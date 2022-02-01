from __future__ import annotations
from os import stat
import uuid
import numpy as np
from uuid import UUID
from typing import Union, Dict, List, Tuple
import copy
import pandas as pd
from mpyc.sectypes import SecInt, SecFxp, SecureNumber
from mpyc.runtime import mpc, Runtime
from psatss.types import sectypes, num_types
from config import PADDED_SIZE, AGG_MINIMUM



def weight_array(
                array, 
                weight: Union[int, float, SecureNumber, Column], 
                runtime: Runtime = mpc
                ) -> List[SecureNumber]:
    """Utility function for weighting array according to array with weights

    :param array: Array to weigh
    :type array: List[SecureNumber]
    :param weight: 
    :type weight: Union[int, float, SecureNumber, Column]
    :param runtime: the mpyc runtime instance, defaults to mpc
    :type runtime: Runtime, optional
    :return: Weighted array
    :rtype: List[SecureNumber]
    """
    if not weight:
        return array
    elif any([isinstance(weight, (int, float, SecureNumber))]):
        return runtime.schur_prod(array, [weight] * len(array))
    elif isinstance(weight, Column):
        return runtime.schur_prod(array, list(map(sectypes["fxp_64"], weight.array)))
    elif isinstance(weight, List):
        return runtime.schur_prod(array, list(map(sectypes["fxp_64"], weight)))

class Column:
    """Base class for column
    """
    def __init__(
                 self, 
                 data: Union[pd.Series, dict]
                 ):
        """Initialize Column object

        :param data: Data for column, along with name for column. Could either be provided by pd.Series, or
                     dictionary where the key represents column name and value is a list of column entries 
        :type data: Union[pd.Series, dict]
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.length = len(data[self.name])
            self.array = list(map(float, data[self.name]))
            self.data = pd.Series(data = data)
        elif isinstance(data, pd.Series):
            self.name = data.name
            self.length = data.shape[0]
            self.data = data
            self.array =  list(map(float, data))
        self.column_id = uuid.uuid4()

    def __repr__(self):
        return self.array.__repr__()

    def __getitem__(self, key):
        return self.array[key]
        
class PrivateColumn:
    """Main object to store private data in secret shared form. 
    These objects are initialized when a pandas.PrivateDataFrame object is created. 
    Columns of the input pandas.DataFrame are turned into a PrivateColumn.
    """
    def __init__(
                 self, 
                 data: pd.Series = None, 
                 sectypes: dict = sectypes, 
                 runtime: Runtime = mpc, 
                 pdf_id: UUID = None,
                 array: List[SecureNumber] = None,
                 mask: List[SecureNumber] = None,
                 name: str = None,
                 padded_size = PADDED_SIZE
                 ):
        """Initialize PrivateColumn. Upon initializing, data array is padded for hiding the size of the input.
        Then, the padded array is secret shared via the Runtime.
        Additionally, some statistical values(max, min, median) for the column are secret shared to prevent expensive
        computations later on.

        :param data: Data for the PrivateColumn
        :type data: pandas.Series
        :param sectypes: Dictionary containing securetypes to be used in secret sharing and MPC, defaults to sectypes
        :type sectypes: dict, optional
        :param runtime: The mpyc.runtime.Runtime object used for SS and MPC, defaults to mpc
        :type runtime: Runtime, optional
        :param pdf_id: UUID for the PrivateDataframe the data belongs to, defaults to None
        :type pdf_id: UUID, optional
        :param array: Secret shared array containing column data
        :type array: List[SecureNumber], optional
        :param mask: Secret shared mask indicating which entries in array contain data and which are padding
        :type mask: List[SecureNumber], optional
        :param name: Name of the column, i.e. the name for the variable it describes
        :type name: str, optional
        :param padded_size: the size of the column array after padding for hiding column length. More padding provides more privacy, but reduces performance
        """
        if array and mask and name:
            self.pdf_id = pdf_id
            self.array = array
            self.runtime = runtime
            self.mask = mask
            self.name = name
            self.max = None
            self.min = None
            self.median = None
            self.padded_size = padded_size
        else:
            self.pdf_id = pdf_id
            self.runtime = runtime
            self.name = data.name
            self.padded_size = padded_size
            try:
                self.array, self.mask = self.pad_and_share(data, padded_size=self.padded_size)
            except TypeError:
                return None
            self.column_id = uuid.uuid4()
            
            self.max = sectypes["fxp_64"](float(data.max()))
            self.min = sectypes["fxp_64"](float(data.min()))
            self.median = sectypes["fxp_64"](float(data.median()))
         
    def pad_and_share(
                      self, 
                      data: pd.Series,
                      padded_size: int = PADDED_SIZE
                      ) -> Tuple[List[SecureNumber], List[SecureNumber]]:
        """Pad and share input array. Padding is necessary for hiding the size of the input,
        which might contain privacy sensitive information. 
        A mask is used to track which values are padding and which aren't.

        :param data: input data for column
        :type data: pd.Series
        :return: Tuple of padded array of secret shared numbers and secret shared mask indicating which values are padding and which aren't
        :rtype: Tuple[List[SecureNumber], List[SecureNumber]]
        """
        array, mask = self.pad(data, padded_size=padded_size)
        array = self.share(array)
        mask = self.share(mask)
        return array, mask
    
    @staticmethod
    def share(
              data: np.array
              )-> List[SecureNumber]:
        """Secret share array

        :param data: Private data to be secret shared
        :type data: np.array
        :raises TypeError: Raised when type is not supported
        :return: Array of secret shared numbers
        :rtype: List[SecureNumber]
        """
        if not any([isinstance(data[0], num_types)]):
            raise TypeError(f"Invalid type for variable {data}, please convert to numerical type.")
        elif isinstance(data[0], np.float64):
            array = list(map(sectypes["fxp_64"], list(map(float, data))))
        elif isinstance(data[0], np.int32):
            array = list(map(sectypes["fxp_64"], list(map(float, data))))
        elif isinstance(data[0], np.int64):
            array = list(map(sectypes["fxp_64"], list(map(float, data))))
        return array
    
    @staticmethod
    def pad(
            data: pd.Series, 
            padded_size: int = PADDED_SIZE
            ) -> List[SecureNumber]:
        """Pad array with random values up to certain size

        :param data: Input array containing data for which input size needs to be hidden
        :type data: pd.Series
        :param padded_size: size of padded array, defaults to PADDED_SIZE
        :type padded_size: int, optional
        :return: Padded array
        :rtype: List[SecureNumber]
        """
        padding = np.random.randint(0, 10, size = padded_size - len(data))
        mask = np.concatenate((np.array([1] * len(data)), np.array([0] * (padded_size - len(data)))), axis=0)
        data = np.concatenate((np.array(data), padding), axis=0)
        return data, mask
    
    def sum(
            self, 
            weight: List[SecureNumber] = None
            ) -> SecureNumber:    
        """Return the sum of all values in this column

        :param weights: Weights for weighting values in sum, defaults to None
        :type weights: List[SecureNumber], optional
        :return: Secret shared number containing result of sum
        :rtype: SecureNumber
        """
        sum = self.runtime.in_prod(self.mask, self.array)
        if weight is not None:
            weighted_sum = sum * weight
            return weighted_sum
        else:
            return sum

    def max(
            self
            ) -> SecureNumber:
        """Return maximum of this column

        :return: Secret maximum of column
        :rtype: SecureNumber
        """
        return self.max

    def min(
            self
            ) -> SecureNumber:
        """Return minimum of column

        :return: Secret minimum of this column
        :rtype: SecureNumber
        """
        return self.min

    def median(
               self
               ) -> SecureNumber:
        """Return median of this column

        :return: Secret shared median of this column
        :rtype: SecureNumber
        """
        return self.median

    def mean(
             self, 
             weight: List[SecureNumber] = None
             ) -> SecureNumber:
        """(Weighted) mean for this column

        :param weight: Optional list of weights, defaults to None
        :type weight: List[SecureNumber], optional
        :return: (Weighted) mean
        :rtype: SecureNumber
        """
        if weight is not None:
            return self.sum(weight) / sum(self.runtime.schur_prod(self.mask, weight * len(self.mask)))
        else:
            return self.sum() / sum(self.mask)

    def __getitem__(
                    self, 
                    key: Union[SecInt, int, slice]
                    ) -> Union[SecureNumber, Union[List[SecureNumber], List[SecureNumber]]]:  
        return self.array[key]
        
    def __repr__(
                self
                ):
        text = "Column containing private data regarding variable '" + str(self.name) #+ "' with " + str(self.length) + " rows."
        return text
    
  
    def __eq__(
                self, 
                other: Union[int, float, PrivateColumn]
                )->List[SecureNumber]:
        """Elementwise equality check for column self and other.

        :param other: [description]
        :type other: Union[int, float, PrivateColumn]
        :return: Secret shared boolean array containing results of elementwise equality check
        :rtype: List[SecureNumber]
        """
        if isinstance(other, PrivateColumn):
            eq = mpc.schur_prod([self.array[i] == other.array[i] for i in range(len(self.array))], self.mask)
        if any([isinstance(other, (float, int))]):
            eq = [self.array[i] == other for i in range(len(self.array))]
        return eq

    def __gt__(
                self, 
                other: Union[int, float, PrivateColumn]
                )->List[SecureNumber]:
        """Elementwise greater than check for column self and other.

        :param other: [description]
        :type other: Union[int, float, PrivateColumn]
        :return: Secret shared boolean array containing results of elementwise greater than check
        :rtype: List[SecureNumber]
        """
        if isinstance(other, PrivateColumn):
            gt = mpc.schur_prod([self.array[i] > other.array[i] for i in range(len(self.array))], self.mask)
        if any([isinstance(other, (float, int))]):
            gt = mpc.schur_prod([self.array[i] > other for i in range(len(self.array))], self.mask)
        return gt
    
    def __lt__(self, other):
        """Elementwise less than check for column self and other.

        :param other: [description]
        :type other: Union[int, float, PrivateColumn]
        :return: Secret shared boolean array containing results of elementwise less than check
        :rtype: List[SecureNumber]
        """
        if isinstance(other, PrivateColumn):
            lt = mpc.schur_prod([self.array[i] < other.array[i] for i in range(len(self.array))], self.mask)
        if any([isinstance(other, (float, int))]):
            lt = mpc.schur_prod([self.array[i] < other for i in range(len(self.array))], self.mask)
        return lt

    def __ge__(self, other):
        """Elementwise greater or equal check for column self and other.

        :param other: [description]
        :type other: Union[int, float, PrivateColumn]
        :return: Secret shared boolean array containing results of elementwise greater or equal check
        :rtype: List[SecureNumber]
        """
        if isinstance(other, PrivateColumn):
            ge = mpc.schur_prod([self.array[i] >= other.array[i] for i in range(len(self.array))], self.mask)
        if any([isinstance(other, (float, int))]):
            ge = mpc.schur_prod([self.array[i] >= other for i in range(len(self.array))], self.mask)
        return ge
        
    def __le__(self, other):
        """Elementwise less or equal check for column self and other.

        :param other: [description]
        :type other: Union[int, float, PrivateColumn]
        :return: Secret shared boolean array containing results of elementwise greater or equal check
        :rtype: List[SecureNumber]
        """
        if isinstance(other, PrivateColumn):
            le = mpc.schur_prod([self.array[i] <= other.array[i] for i in range(len(self.array))], self.mask)
        if any([isinstance(other, (float, int))]):
            le = mpc.schur_prod([self.array[i] <= other for i in range(len(self.array))], self.mask)
        return le

class EmptyColumn(Column):
    def __init__(self, data, pdf_id):
        super().__init__(data)
        self.array = ["NaN"]*self.length
        self.pdf_id = pdf_id



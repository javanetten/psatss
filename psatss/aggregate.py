from psatss.columns import PrivateColumn, weight_array
from psatss.types import sectypes
from psatss.config import AGG_MINIMUM, PADDED_SIZE

from __future__ import annotations
import math
from pandas.core.indexing import convert_to_index_sliceable
from privatedataframe import PrivateDataFrame
from typing import List, Union, Tuple, Dict
from uuid import UUID
from mpyc.sectypes import SecureNumber
from mpyc.runtime import mpc, Runtime



class Aggregate:
    """Class representing Aggregate of datasets. 
    This object is the interface between a statistician who wants to produce aggregate statistics and private respondent-level data.
    Building blocks of the Aggregate are AggregatedColumn's. 
    These object provide methods for conducting statistics on the aggregate in a privacy-friendly way, 
    """
    def __init__(       
                 self, 
                 pdfs: List[PrivateDataFrame] = None,
                 agg_cols: List[AggregatedColumn] = None,
                 aggregate_minimum: int = AGG_MINIMUM
                 ):
        """Initialize aggregate object either from a list of PrivateDataFrame's or a list of AggregateColumns.
        The former occurs when the data scientist aggregates private datasets, 
        the latter when an Aggregate is instantiated from within another Aggregate.

        :param pdfs: PrivateDataFrame's containing the data to be aggregated, defaults to None
        :type pdfs: List[PrivateDataFrame], optional
        :param agg_cols: AggregatedColumns that form Aggregate, defaults to None
        :type agg_cols: List[AggregatedColumn], optional
        :param aggregate_minimum: Minimum of datasets that need to be aggregated for reconstruction to be allowed, defaults to AGG_MINIMUM
        :type aggregate_minimum: int, optional
        """
        if agg_cols:
            self.agg_cols = agg_cols
            self.no_of_sources = agg_cols[0].runtime.run(agg_cols[0].runtime.output(agg_cols[0].no_of_sources))
            self.private_data = {}
            self.pdf_ids = agg_cols[0].pdf_ids
            self.column_names = []
            self.cols_by_pdf = {pdf_id: {} for pdf_id in self.pdf_ids}
            for agg_col in agg_cols:
                self.private_data.update({agg_col.name: agg_col})
                self.pdf_ids = agg_col.pdf_ids
                self.column_names.append(agg_col.name)
                self.aggregate_minimum = agg_col.aggregate_minimum
                for col in agg_col.data:
                    self.cols_by_pdf[col.pdf_id].update({col.name: col})
        else:
            self.pdfs = pdfs
            self.no_of_sources = len(pdfs)
            self.column_names = pdfs[0].column_names
            self.pdf_ids = [pdf.pdf_id for pdf in pdfs]
            self.aggregate_minimum = aggregate_minimum

            self.private_data = {}
            for column in pdfs[0].private_data.keys():
                self.private_data.update(
                                         {
                                            column: AggregatedColumn(
                                                                    cols = [pdf[column] for pdf in pdfs], 
                                                                    aggregate_minimum = self.aggregate_minimum
                                                                    )
                                         }
                                        )

            self.weight = {}
            for weight in pdfs[0].weight:
                self.weight.update(
                                    {
                                        weight: {pdf.pdf_id: pdf.weight[weight] for pdf in pdfs}
                                    }
                                    )

            self.cols_by_pdf = {pdf_id: {} for pdf_id in self.pdf_ids}
            for agg_col in self.private_data.values():
                for col in agg_col.data:
                    self.cols_by_pdf[col.pdf_id].update({col.name: col})

        

    def __getitem__(
                    self,
                    which: Union[str, List[Union[PrivateColumn, List[SecureNumber]]]]
                    ):
        
        if isinstance(which, str):
            if which in self.column_names:
                return self.private_data[which]
        if isinstance(which, list):
            #We enter this branch when we call for example agg[agg["col_a"]==agg["col_b"]]. 
            #We should then return a new aggregate satisfying the condition.
            agcs = [] #List of AggregatedColumns that will make up the new Aggregate
            for column_name, agg_col in self.private_data.items():
                pcs = [] #PrivateColumns that make up this AggregatedColumn
                for col, mask in which:
                    pdf_id = col.pdf_id
                    column = self.cols_by_pdf[pdf_id][column_name]
                    new_mask = mask
                    pcs.append(PrivateColumn(array=column.array, mask=new_mask, pdf_id=pdf_id, name=column.name))
                agc = AggregatedColumn(cols=pcs)
                agcs.append(agc)
            return Aggregate(agg_cols = agcs, aggregate_minimum = self.aggregate_minimum)
    def __repr__(self):
        msg = "Aggregate of " + str(self.no_of_sources) + " datasets\nPrivate Variables:\t" + str(list(self.private_data.keys()))
        return msg

class AggregatedColumn():
    """Class representing a collection(aggregate) of PrivateColumns. 
    This class provides methods for computing aggregate functions, as well as
    privately selecting subsets based on comparison operators
    """
    def __init__(
                 self, 
                 data: List[PrivateColumn] = None, 
                 pdf_ids: List[UUID] = None, 
                 runtime: Runtime = mpc,
                 cols: List[PrivateColumn] = None,
                 aggregate_minimum = AGG_MINIMUM
                 ):
        """Initialize AggregatedColumn. 
        These objects are instantiated when an Aggregate is instantiated
        from a list of PrivateDataFrames.

        :param data: :class:PrivateColumn 's to aggregate
        :type data: List[PrivateColumn]
        :param pdf_ids: ID's of the :class:PrivateDataframe 's from which the data originates, defaults to None
        :type pdf_ids: List[UUID], optional
        :param runtime: mpyc runtime object, used for MPC protocols, defaults to mpc
        :type runtime: Runtime, optional
        """

        self.aggregate_minimum = aggregate_minimum
        
        if cols:
            self.name = cols[0].name
            self.data = cols
            self.pdf_ids = [col.pdf_id for col in cols]
            self.runtime = cols[0].runtime
            self.no_of_sources = sum([sum(col.mask) > 0  for col in cols])
            if self.runtime.run(self.runtime.output(self.no_of_sources)) >= self.aggregate_minimum:
                self.suff_agg = True
            else:
                self.suff_agg = False
        else: 
            self.name = data[0].name
            self.data = data #All columns to be aggregated should describe the same variable
            self.pdf_ids = pdf_ids
            self.runtime = runtime
            self.no_of_sources = len(data)
            if self.no_of_sources >= self.aggregate_minimum:
                self.suff_agg = True
            else:
                self.suff_agg = False
        
        self.cols_by_pdf = {pdf_id: None for pdf_id in self.pdf_ids}
        for col in self.data:
            self.cols_by_pdf[col.pdf_id] = col
        
        self.N = sum([sum(column.mask) for column in self.data])
        


    def sum(
            self, 
            weight: Dict[UUID, List[SecureNumber]] = None, 
            reconstruct: bool = False
            ) -> Union[SecureNumber, float]:
        """Compute (weighted) sum 

        :param weight: [description], defaults to None
        :type weight: Dict[UUID, List[SecureNumber]], optional
        :param reconstruct: Boolean flag, defaults to False
        :type reconstruct: bool, optional
        :return: (weighted) sum of all aggregated columns
        :rtype: Union[SecureNumber, float]
        """
        if weight is not None:
            result = sum([column.sum(weight = weight[column.pdf_id]) for column in self.data])
        else:
            result = sum([column.sum() for column in self.data])
        if reconstruct:
            return self.reconstruct(result)
        return result

    def mean(
            self, 
            weight: dict = None,
            reconstruct = False
            ) -> Union[SecureNumber, float]:
        """Compute (weighted) mean for aggregate of columns

        :param weight: dictionary storing weight by pdf_id, defaults to None
        :type weight: dict, optional
        :param reconstruct: boolean flag, defaults to False
        :type reconstruct: bool, optional
        :return: Mean for aggregated column
        :rtype: Union[SecureNumber, float]
        """
        if weight is not None:
            s = self.sum(weight = weight)
            #compute sum of weight
            s_w_by_pdf = []
            for pdf_id, weight_array in weight.items():
                col = self.cols_by_pdf[pdf_id]
                s_w_by_pdf.append(sum(self.runtime.schur_prod(col.mask, weight_array)))
            s_w = sum(s_w_by_pdf)
            result = s / s_w
        else:
            s = self.sum()
            result =  s / self.N
        if reconstruct:
            return self.reconstruct(result)
        return result
    
    def variance(
                 self, 
                 reconstruct = False, 
                 ddof: int = 1
                 ) -> Union[SecureNumber, float]:
        """Compute variance for aggregated column

        :param reconstruct: boolean flag, defaults to False
        :type reconstruct: bool, optional
        :param ddof: delta degrees of freedom. Use 1 for sample variance, 0 for population variance, defaults to 1
        :type ddof: int, optional
        :return: variance for aggregate
        :rtype: Union[SecureNumber, float]
        """
        mean = self.mean()
        devs = [self.runtime.schur_prod(self.runtime.vector_sub(X.array, [mean] * len(X.array)), X.mask) for X in self.data]
        flat_devs = [value for dev in devs for value in dev]
        variance = self.runtime.in_prod(flat_devs, flat_devs) / (self.N - ddof)
        if reconstruct:
            return self.reconstruct(variance)
        return variance
    
    def cov(
            self, 
            other: AggregatedColumn,
            ddof: int = 1,
            reconstruct = True
            ) -> Union[SecureNumber, float]:
        """Compute covariance between two aggregated columns as
        cov(X,Y) = sum_i((X-mean_X)(Y-mean_Y))_i / N_X - ddof

        :param other: Other aggregated column to compute covariance with
        :type other: AggregatedColumn
        :param ddof: delta degrees of freedom. Use 1 for sample covariance, 0 for population covariance, defaults to 1
        :type ddof: int, optional
        :param reconstruct: boolean flag, defaults to True
        :type reconstruct: bool, optional
        :return: covariance, either secret shared or reconstructed
        :rtype: Union[SecureNumber, float]
        """
        mean_X = self.mean()
        mean_Y = other.mean()

        devs_X = [self.runtime.schur_prod(self.runtime.vector_sub(X.array, [mean_X] * len(X.array)), X.mask) for X in self.data]
        devs_Y = [self.runtime.schur_prod(self.runtime.vector_sub(Y.array, [mean_Y] * len(Y.array)), Y.mask) for Y in other.data]

        flat_devs_X = [value for dev in devs_X for value in dev]
        flat_devs_Y = [value for dev in devs_Y for value in dev]

        cov_XY = self.runtime.in_prod(flat_devs_X, flat_devs_Y) / (self.N - ddof)

        if reconstruct:
            return self.reconstruct(cov_XY)
        
        return cov_XY

    def max(
            self,
            reconstruct: bool = False
            ) -> Union[SecureNumber, float]:
        """Compute maximum of aggregated column

        :param reconstruct: boolean flag for reconstruction, defaults to False
        :type reconstruct: bool, optional
        :return: Maximum, either secret shared or reconstructed
        :rtype: Union[SecureNumber, float]
        """
        max = mpc.max(X.max() for X in self.data)
        if reconstruct:
            max = self.reconstruct(max)
        return max

    def min(
            self,
            reconstruct: bool = False
            ) -> Union[SecureNumber, float]:
        """Compute minimum of aggregated column

        :param reconstruct: boolean flag for reconstruction, defaults to False
        :type reconstruct: bool, optional
        :return: minimum of aggregated column
        :rtype: SecureNumber, float
        """
        min = mpc.min(X.min() for X in self.data)
        if reconstruct:
            min = self.reconstruct(min)
        return min

    def __eq__(
                self, 
                other: Union[int, float, AggregatedColumn]
                ) -> List[Tuple[PrivateColumn, SecureNumber]]:
        """Elementwise equality check for aggregated column, similar to the one for PrivateColumn

        :param other: Value or Column to compare with
        :type other: Union[int, float, AggregatedColumn]
        :return: List of boolean arrays(masks) containing result of elementwise comparison
        :rtype: List[Tuple[PrivateColumn, SecureNumber]]
        """
        if isinstance(other, AggregatedColumn):
            cols = self.cols_by_pdf
            other_cols = other.cols_by_pdf
            masks = [(cols[pdf_id], cols[pdf_id] == other_cols[pdf_id]) for pdf_id in self.pdf_ids]
        else:
            masks = [(column, column == other) for column in self.data]
        return masks

    def __gt__(
                self, 
                other: Union[int, float, AggregatedColumn]
                ) -> List[Tuple[PrivateColumn, SecureNumber]]:
        """Elementwise greater than check for aggregated column, similar to the one for PrivateColumn

        :param other: Value or Column to compare with
        :type other: Union[int, float, AggregatedColumn]
        :return: List of boolean arrays(masks) containing result of elementwise comparison
        :rtype: List[Tuple[PrivateColumn, SecureNumber]]
        """
        if isinstance(other, AggregatedColumn):
            cols = self.cols_by_pdf
            other_cols = other.cols_by_pdf
            masks = [(cols[pdf_id], cols[pdf_id] > other_cols[pdf_id]) for pdf_id in self.pdf_ids]
        else:
            masks = [(column, column > other) for column in self.data]
        return masks

    def __lt__(
                self, 
                other: Union[int, float, AggregatedColumn]
                ) -> List[Tuple[PrivateColumn, SecureNumber]]:
        """Elementwise less than check for aggregated column, similar to the one for PrivateColumn

        :param other: Value or Column to compare with
        :type other: Union[int, float, AggregatedColumn]
        :return: List of boolean arrays(masks) containing result of elementwise comparison
        :rtype: List[Tuple[PrivateColumn, SecureNumber]]
        """
        if isinstance(other, AggregatedColumn):
            cols = self.cols_by_pdf
            other_cols = other.cols_by_pdf
            masks = [(cols[pdf_id], cols[pdf_id] < other_cols[pdf_id]) for pdf_id in self.pdf_ids]
        else:
            masks = [(column, column < other) for column in self.data]
        return masks

    def __le__(
                self, 
                other: Union[int, float, AggregatedColumn]
                ) -> List[Tuple[PrivateColumn, SecureNumber]]:
        """Elementwise less or equal check for aggregated column, similar to the one for PrivateColumn

        :param other: Value or Column to compare with
        :type other: Union[int, float, AggregatedColumn]
        :return: List of boolean arrays(masks) containing result of elementwise comparison
        :rtype: List[Tuple[PrivateColumn, SecureNumber]]
        """
        if isinstance(other, AggregatedColumn):
            cols = self.cols_by_pdf
            other_cols = other.cols_by_pdf
            masks = [(cols[pdf_id], cols[pdf_id] <= other_cols[pdf_id]) for pdf_id in self.pdf_ids]
        else:
            masks = [(column, column <= other) for column in self.data]
        return masks

    def __ge__(
                self, 
                other: Union[int, float, AggregatedColumn]
                ) -> List[Tuple[PrivateColumn, SecureNumber]]:
        """Elementwise greater or equal check for aggregated column, similar to the one for PrivateColumn

        :param other: Value or Column to compare with
        :type other: Union[int, float, AggregatedColumn]
        :return: List of boolean arrays(masks) containing result of elementwise comparison
        :rtype: List[Tuple[PrivateColumn, SecureNumber]]
        """
        if isinstance(other, AggregatedColumn):
            cols = self.cols_by_pdf
            other_cols = other.cols_by_pdf
            masks = [(cols[pdf_id], cols[pdf_id] >= other_cols[pdf_id]) for pdf_id in self.pdf_ids]
        else:
            masks = [(column, column >= other) for column in self.data]
        return masks

    def __repr__(self):
        msg = "Aggregated column containing data from " + str(len(self.data)) + " datasets regarding variable " + '"' + str(self.name) + '"'
        return msg

    def reconstruct(self, r):
        if self.suff_agg:
            return mpc.run(mpc.output(r))
        else:
            raise ReconstructionError('Aggregation level insufficient')


class ReconstructionError(Exception):
    """Exception class for illegal reconstruction

    :param Exception: Raised when an reconstruction is called but number of datasets in the aggregate does not meet the aggregation minimum
    :type Exception: Exception
    """
    pass

    
    
    


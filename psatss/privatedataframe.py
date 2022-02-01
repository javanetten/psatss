from psatss.types import num_types
from psatss.columns import Column, PrivateColumn, PrivateColumn, EmptyColumn
from psatss.config import PADDED_SIZE

import pandas as pd
import uuid
from typing import List, Union
from psatss.types import sectypes
from mpyc.runtime import mpc
import copy
import numpy as np

class PrivateDataFrame():

    def __init__(
                self, 
                df: pd.DataFrame, 
                description: str = None,
                sectypes = sectypes,
                runtime = mpc,
                padded_size=PADDED_SIZE
                ):
        """PrivateDataFrame objects are created from DataFrames by, on a column by column basis,
        padding the columns for hiding the size and secret sharing the padded column.

        :param df: DataFrame containing private data that needs to be hidden
        :type df: pd.DataFrame
        :param description: Description for the DataFrame, defaults to None
        :type description: str, optional
        :param sectypes: secure datatypes necessary for secret sharing, defaults to sectypes
        :type sectypes: List[SecType], optional
        :param runtime: :class mpyc.Runtime: object necessary for secret sharing and running mpc protocols, defaults to mpc
        :type runtime: :class mpyc.Runtime:, optional
        :return: The PrivateDataFrame
        :rtype: :PrivateDataFrame
        """
        self.runtime = runtime
        self.column_names = list(df.columns)
        self.pdf_id = uuid.uuid4()
        if description:
            self.description = description
        else:
            self.description = "No description provided"    

        if df.shape[0] == 0:
            print("Dataframe is empty")
            return None
          
        self.private_variables = []

        #Create Dictionaries representing private and public data
        self.private_data = {}
        self.empty_columns = {}
        self.non_num_columns = []
        #define size for padded columns
        self.padded_size = padded_size


        for variable in self.column_names:
            if not df[variable].dtype in num_types:
                print("Skipping column " + variable + " because datatype is not supported")
                self.non_num_columns.append(variable)
            elif any(df[variable].isnull()):
                self.empty_columns.update(
                                         {
                                             variable: EmptyColumn(
                                                                df[variable],
                                                                pdf_id = self.pdf_id
                                                                )
                                          }
                                        )
            else:
                self.private_data.update(
                                        {
                                            variable: PrivateColumn(
                                                                    data = df[variable], 
                                                                    sectypes=sectypes,
                                                                    runtime=runtime, 
                                                                    pdf_id = self.pdf_id,
                                                                    padded_size=self.padded_size
                                                                    )
                                        }
                                        )
                self.private_variables.append(variable)
        
        self.weight = {}

    def __repr__(self):
        text = "Private Dataframe containing data regarding variables " + \
             str([column_name for column_name in self.private_variables])
        if len(list(self.non_num_columns)) > 0:
            text = text + "\n Variables " + str([column_name for column_name in self.non_num_columns]) + " were skipped because the contents were non numerical. These have to be encoded as numbers for secret sharing"
        if len(list(self.empty_columns.items())) > 0:
            text = text + "\n Columns " + str([column_name for column_name in self.empty_columns.keys()]) + "contained missing values. Consider imputing these before sharing"
        return text
    
    def __getitem__(self, which):
        return self.private_data[which]
    
    def add_public_weight(self, name, weight: Union[int, float]):
        self.weight.update({name: sectypes['fxp_64'](weight)})
        
        
        #length = self.padded_size
        #weight = PrivateColumn.share(data=np.array([weight] * length))
        #self.weight.update({name: weight})

if __name__=='__main__':
    df_1 = pd.DataFrame({"a": ['a','b','c'], "b": [1,3,5]})
    pdf = PrivateDataFrame(df_1)
    pdf

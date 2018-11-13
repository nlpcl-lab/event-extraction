import tensorflow as tf
from Util import train_parser
from Model import Model as ARGUMENT_MODEL
from Model import Model as TRIGGER_MODEL
from Dataset import Dataset as ARGUMENT_DATASET
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET

from Util import train_parser
if __name__=='__main__':
    task, subtask = train_parser()
    subtask_type = 'IDENTIFICATION' if subtask==1 else 'CLASSIFICATION'


===============================
SPACE-IDEAS: A Dataset for Salient Information Detection in Space Innovation
===============================

|License| |License2| |PyPI pyversions|

Developed with 💛 at `Expert.ai Research Lab <https://github.com/expertailab>`__

-  License: Apache 2.0 (Software) and Attribution 4.0 International (Datasets)
-  Paper: TBD

Content
---------------
This repository contains **code** and **datasets** for the paper titled *SPACE-IDEAS: A Dataset for Salient Information Detection in Space Innovation*. It is organized as follows:

- `data/processed <data/processed>`_: Contains the SPACE-IDEAS and SPACE-IDEAS+ datasets.
- The rest of the repository contains the code for conducting paper experiments.

Installation
------------

The whole project is handled with ``make``, go to a terminal an issue:

.. code:: bash

   git clone https://github.com/expertailab/SPACE-IDEAS
   cd SPACE-IDEAS
   make setup
   conda activate ideas_annotation
   make install-as-pkg

Reproducibility
---------------
**Data split:**
To split the SPACE-IDEAS dataset in train and test splits, we can run the split_data.py script:

.. code:: bash

   python scripts/split_data.py

Two files, train.jsonl and test.jsonl, will be created in the data/processed folder.

**Single-sentence classification:**

To train a single sentence classifier using the training SPACE-IDEAS data without context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl

If we want to use the context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context

To train using the SPACE-IDEAS plus dataset, we have to change the input_train_dataset to :

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/space-ideas_plus.jsonl --input_test_dataset data/processed/test.jsonl --use_context

**Sequential sentence classification:**

We need to split the train set in train2 and dev set, we can do this with:

.. code:: bash

   python scripts/split_train_data.py

Two files, train2.jsonl and dev.jsonl, will be created in the data/processed folder. 

We clone the sequential_sentence_classification repository, create a new conda environment and install the required allennlp library.

.. code:: bash

   git clone https://github.com/expertailab/sequential_sentence_classification.git
   cd sequential_sentence_classification/
   git checkout allennlp2
   conda create -n sequential_sentence_classification python=3.9
   conda activate sequential_sentence_classification
   pip install allennlp==2.0.0

We have to modify the train.sh script in scripts folder, with the data paths:

.. code:: bash

   TRAIN_PATH=../data/processed/train2.jsonl
   DEV_PATH=../data/processed/dev.jsonl
   TEST_PATH=../data/processed/test.jsonl

We can now run the trainining stript with:

.. code:: bash

   ./scripts/train.sh tmp_output_dir_space-ideas

The trained model will be at tmp_output_dir_space-ideas/model.tar.gz, we can get the test predictions with:

.. code:: bash

   python -m allennlp predict tmp_output_dir_space-ideas/model.tar.gz ../data/processed/test.jsonl --include-package sequential_sentence_classification --predictor SeqClassificationPredictor --cuda-device 0 --output-file space-ideas-predictions.json
   
Now we can obtain the prediction metrics with:

.. code:: bash

   cd ..
   conda activate ideas_annotation
   python scripts/sequential_sentence_classification_metrics.py --prediction_test_file sequential_sentence_classification/space-ideas-predictions.json --gold_test_file data/processed/test.jsonl

Sequential Transfer Learning
~~~~~~~~~~~~~~~~~~~~~
**Single-sentence classification:**

We can train a model, using for example SPACE-IDEAS plus dataset, and use that trained model to finetune on the SPACE-IDEAS dataset, we can do this with the following command:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --model $PATH_TO_TRAINED_MODEL --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context

**Sequential sentence classification:**

First we need to train a model using the SPACE-IDEAS plus dataset, we can do it by changing the TRAIN_PATH variable in the train.sh script and point to the dataset location (../data/processed/space-ideas_plus.jsonl). Then we launch the training with:

.. code:: bash

   cd sequential_sentence_classification/
   conda activate sequential_sentence_classification
   ./scripts/train.sh tmp_output_dir_space-ideas-plus

When the training is finished, we will have a model.tar.gz file in the "tmp_output_dir_space-ideas-plus" folder. To finally train using the SPACE-IDEAS dataset, we need to change the "config.jsonnet" file in the "sequential_sentence_classification" folder, we need to change the "model" field in line 40, to the following:

.. code-block:: json

   ..
   "model": {
      "type": "from_archive",
      "archive_file": "tmp_output_dir_space-ideas-plus/model.tar.gz"
   },
   ..
Then we change again the TRAIN_PATH variable in the train.sh script to point to the dataset location (../data/processed/train2.jsonl), and launch the training with:

.. code:: bash

   ./scripts/train.sh tmp_output_dir_space-ideas_from_space-ideas-plus

The trained model will be at tmp_output_dir_space-ideas_from_space-ideas-plus/model.tar.gz, we can get the test predictions with:

.. code:: bash

   python -m allennlp predict tmp_output_dir_space-ideas_from_space-ideas-plus/model.tar.gz ../data/processed/test.jsonl --include-package sequential_sentence_classification --predictor SeqClassificationPredictor --cuda-device 0 --output-file space-ideas-predictions_from_space-ideas-plus.json

Now we can obtain the prediction metrics with:

.. code:: bash

   cd ..
   conda activate ideas_annotation
   python scripts/sequential_sentence_classification_metrics.py --prediction_test_file sequential_sentence_classification/space-ideas-predictions_from_space-ideas-plus.json --gold_test_file data/processed/test.jsonl

Multi-Task Learning
~~~~~~~~~~~~~~~~~~~~~
**Single-sentence classification:**

By deafult, we can do multitask training using all the available datasets (SPACE-IDEAS, SPACE-IDEAS plus) with:

.. code:: bash

   python scripts/merge_space-ideas_dataset.py
   python ideas_annotation/modeling/idea_dataset_multitask_sentence_classification.py

**Sequential sentence classification:**

To run the multitask traininig with sequential sentence classification, we need to install a variation of the `grouphug <https://github.com/sanderland/grouphug>`_ library. We can install it with:

.. code:: bash

   git clone https://github.com/expertailab/grouphug.git
   cd grouphug
   pip install .
   cd ..

Now we can run the idea_dataset_multitask_sentence_classification.py script:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_multitask_sentence_classification.py

In line 135 of the script, we can set the combinations of datasets that we want to train: ["chatgpt", "gold"].

How to cite
-----------

To cite this research please use the following: `TBD`


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/

|Expert.ai favicon| Expert.ai
-----------------------------

At Expert.ai we turn language into data so humans can make better
decisions. Take a look `here <https://expert.ai>`__!

.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |License2| image:: https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by/4.0/
.. |Expert.ai favicon| image:: https://www.expert.ai/wp-content/uploads/2020/09/favicon-1.png
   :target: https://expert.ai

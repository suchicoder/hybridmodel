# hybridmodel
This code aims to extract class diagram elements from requirement documents using annotation data. The process involves several steps, including text preprocessing, feature extraction, model training, and result merging.
Steps:

    Text Preprocessing:
        The requirement text is split according to the use cases to facilitate further processing.

    BERT Embedding:
        The split text is fed into a BERT model to obtain word embeddings, which capture semantic information from the text.

    Clustering:
        The word embeddings are used for clustering to group similar words together, ultimately forming clusters.

    Centroid Vector Calculation:
        Centroid vectors are computed for each cluster, representing the central point of the cluster.

    Rule-Based Feature Creation:
        Rule-based features are generated based on predefined rules to capture specific patterns or characteristics in the text.

    Word-Based Feature Creation:
        Features based on individual words are extracted to provide additional information for the classification task.

    CRF Model Training:
        The features extracted from the previous steps are fed into a Conditional Random Field (CRF) model for training.

    Result Merging:
        Finally, the results from different class diagram elements are merged to produce the final output.

Usage:

    Ensure that all necessary libraries and dependencies are installed.
    Run the script finalcode.py to execute the experiment.
    Provide the path to the annotation data and requirement documents as input parameters.
    The output will include the extracted class diagram elements.

Files:

    finalcoded.py: Main script to execute the experiment.
    split.py: Contains functions for requirement splitting.
    vectorfile.py: Functions for feature extraction including BERT embedding, clustering, and feature creation.
    merge.py: Code for merging the extracted class diagram elements.
    relation.py: Code for merging the extracting relation between classes.
    Rule.py: Code for extracting rule based features.

Note:

    Ensure that the annotation data and requirement documents are appropriately formatted.
    Feel free to contact the authors for any questions or assistance.

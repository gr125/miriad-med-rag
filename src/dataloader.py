import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_dataset(dir):
    """
    Reads a directory of parquet files into a pyarrow Dataset.
    """
    return pq.ParquetDataset(dir)

def split_dataset(dataset, output_dir, stratify_column='specialty', test_size=0.1, val_size=0.1):
    """
    Splits a pyarrow Dataset into train, validation, and test sets,
    stratified by a given column, and saves them as parquet files.
    This version avoids loading the entire dataset into a pandas DataFrame.

    Args:
        dataset (pyarrow.parquet.ParquetDataset): The dataset to split.
        output_dir (str): The directory to save the output files.
        stratify_column (str): The column to stratify by.
        test_size (float): The proportion of the dataset to allocate to the test set.
        val_size (float): The proportion of the dataset to allocate to the validation set.
    """
    print("Reading table from dataset...")
    table = dataset.read()
    print(f"Table created with {len(table)} rows.")
    table.cast(pa.schema([pa.field('qa_id', pa.string()),
                          pa.field('paper_id', pa.int64()),
                          pa.field('question', pa.string()),
                          pa.field('answer', pa.large_string()),
                          pa.field('paper_url', pa.string()),
                          pa.field('paper_title', pa.string()),
                          pa.field('passage_text', pa.large_string()),
                          pa.field('passage_position', pa.int8()),
                          pa.field('year', pa.int64()),
                          pa.field('venue', pa.string()),
                          pa.field('specialty', pa.string())]))
    os.makedirs(output_dir, exist_ok=True)

    # Drop rows with nulls in the stratification column.
    print(f"Filtering null values in qa_id, question, passage_text, '{stratify_column}' column...")
    filter_mask = pc.and_(
        pc.and_(
            pc.invert(pc.is_null(table["qa_id"])),
            pc.invert(pc.is_null(table["question"]))),
        pc.and_( 
            pc.invert(pc.is_null(table["passage_text"])),
            pc.invert(pc.is_null(table[stratify_column]))
        )
    )
    table = table.filter(filter_mask)
    print(f"Table filtered, {len(table)} rows remaining.")

    print("Identifying classes with a single sample...")
    # Get value counts of the stratification column
    value_counts = table[stratify_column].value_counts()

    # Find classes with only one sample
    single_sample_classes = value_counts.filter(pc.equal(value_counts.field(1), 1))
    single_sample_class_names = single_sample_classes.field(0)
    
    if len(single_sample_class_names) > 0:
        print(f"Found {len(single_sample_class_names)} classes with a single sample. These will be added to the training set.")
        
        # Create a mask for rows that are in the single sample classes
        single_sample_mask = pc.is_in(table[stratify_column], value_set=single_sample_class_names)
        
        # Split the table into single-sample and multi-sample tables
        single_sample_table = table.filter(single_sample_mask)
        multi_sample_table = table.filter(pc.invert(single_sample_mask))
        
        print(f"Single sample table size: {len(single_sample_table)}")
        print(f"Multi sample table size: {len(multi_sample_table)}")
    else:
        print("No classes with a single sample found.")
        multi_sample_table = table
        single_sample_table = None
    
    print("Preparing for stratified split on the remaining data...")
    # Get the stratification column as a numpy array for sklearn from the multi_sample_table
    stratify_data = multi_sample_table[stratify_column].to_numpy()
    indices = np.arange(len(multi_sample_table))

    # Calculate split sizes
    train_size = 1 - test_size - val_size

    # First split for test set
    print(f"Splitting data. Test size: {test_size}")
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_data,
        random_state=42
    )

    # Second split for validation set
    relative_val_size = val_size / (train_size + val_size)
    # We need to stratify on the subset of the original stratify_data
    train_val_stratify = stratify_data[train_val_indices]
    print(f"Splitting remaining data for train/validation. Relative validation size: {relative_val_size}")
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        stratify=train_val_stratify,
        random_state=42
    )

    print("Creating table splits from indices...")
    # Use the indices to create the final tables from multi_sample_table
    train_table_split = multi_sample_table.take(pa.array(train_indices))
    val_table = multi_sample_table.take(pa.array(val_indices))
    test_table = multi_sample_table.take(pa.array(test_indices))

    # Combine the single sample table with the training table
    if single_sample_table is not None:
        print("Concatenating single-sample table with the training set.")
        train_table = pa.concat_tables([train_table_split, single_sample_table])
    else:
        train_table = train_table_split

    # Save to parquet files
    print(f"Saving split files to {output_dir}...")
    pq.write_table(train_table, os.path.join(output_dir, 'train.parquet'))
    pq.write_table(val_table, os.path.join(output_dir, 'val.parquet'))
    pq.write_table(test_table, os.path.join(output_dir, 'test.parquet'))

    print(f"Train set size: {len(train_table)}")
    print(f"Validation set size: {len(val_table)}")
    print(f"Test set size: {len(test_table)}")
    print(f"Output files saved in {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load and split the MIRIAD dataset.")
    parser.add_argument('--data_dir', type=str, default='data/miriad-5.8M/data', help='Directory containing the raw parquet files.')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Directory to save the split parquet files.')
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_dataset(args.data_dir)
    print("Dataset loaded.")

    print("Splitting dataset...")
    split_dataset(dataset, args.output_dir)
    print("Dataset splitting complete.")
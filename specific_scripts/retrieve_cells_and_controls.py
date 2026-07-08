"""
=============================================================================
RETRIEVE CELLS AND CONTROLS USING CELLTYPE INDEX
=============================================================================

This script uses the prebuilt celltype index to:
1. Query for cells of a specific type (semantic search on query)
2. Retrieve n cells of that type
3. Retrieve n control cells NOT of that type
4. Save both in geneformer-compatible format

Geneformer format expects:
- CSV with columns: [cell_type, gene_name, expression_value]
- Each row is one cell's expression for one gene
- Can be split into separate files for case/control
"""

import os
import sys
import argparse
import numpy as np
import pickle
import hnswlib
import json
import pandas as pd
from google.genai import Client
from cellxgene_census import open_soma
import cellxgene_census

def load_celltype_index(index_dir: str, verbose: bool = True) -> tuple:
    """
    Load prebuilt celltype index and metadata.
    
    Parameters
    ----------
    index_dir : str
        Directory containing the index files
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    tuple
        (celltype_list, embeddings_array, hnswlib_index)
    """
    
    if verbose:
        print(f"\n📂 Loading prebuilt celltype index from {index_dir}...")
    
    # Load cell type list
    celltype_list_path = os.path.join(index_dir, "celltype_list.pkl")
    with open(celltype_list_path, 'rb') as f:
        celltype_list = pickle.load(f)
    if verbose:
        print(f"✅ Loaded {len(celltype_list)} cell types")
    
    # Load embeddings
    embeddings_path = os.path.join(index_dir, "celltype_embeddings.npy")
    embeddings_array = np.load(embeddings_path)
    if verbose:
        print(f"✅ Loaded embeddings: {embeddings_array.shape}")
    
    # Load hnswlib index
    index_path = os.path.join(index_dir, "celltype_index.bin")
    dimension = embeddings_array.shape[1]
    index = hnswlib.Index(space='l2', dim=dimension)
    index.load_index(index_path)
    if verbose:
        print(f"✅ Loaded hnswlib index with {index.get_current_count()} cell types")
    
    return celltype_list, embeddings_array, index


def retrieve_cells_and_controls(
    cell_type_query: str,
    num_cells: int = 1000,
    num_controls: int = 1000,
    output_dir: str = "./geneformer_data",
    output_prefix: str = "cells_vs_controls",
    api_key: str = None,
    index_dir: str = None,
    create_splits: bool = False,
    split_fractions: dict = None,
    verbose: bool = True
) -> dict:
    """
    Query for a cell type, retrieve cells and controls, save as geneformer format.
    
    Can optionally split into train/test/validation sets.
    
    Parameters
    ----------
    cell_type_query : str
        Natural language query for cell type (e.g., "hepatocyte", "T cell", "immune cells")
    num_cells : int, optional
        Number of cells of the target type to retrieve (default: 1000)
        If create_splits=True, this is the total that will be split
    num_controls : int, optional
        Number of control cells (not of target type) to retrieve (default: 1000)
        If create_splits=True, this is the total that will be split
    output_dir : str, optional
        Directory to save output files (default: ./geneformer_data)
    output_prefix : str, optional
        Prefix for output files (default: cells_vs_controls)
    api_key : str, optional
        Google API key (default: GOOGLE_API_KEY env var)
    index_dir : str, optional
        Directory containing prebuilt index (default: ./celltype_index)
    create_splits : bool, optional
        If True, split data into train/test/validation sets (default: False)
    split_fractions : dict, optional
        Dictionary with split fractions: {'train': 0.7, 'test': 0.15, 'val': 0.15}
        Default: {'train': 0.7, 'test': 0.15, 'val': 0.15}
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    dict
        Summary of retrieved data with file paths
    """
    
    # Default split fractions
    if split_fractions is None:
        split_fractions = {'train': 0.7, 'test': 0.15, 'val': 0.15}
    
    # Validate split fractions
    if create_splits:
        total_fraction = sum(split_fractions.values())
        if not (0.99 < total_fraction < 1.01):  # Allow small floating point errors
            raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction}")
    
    # Setup
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Set GOOGLE_API_KEY env var or pass --api-key")
    
    index_dir = index_dir or "./celltype_index"
    os.makedirs(output_dir, exist_ok=True)
    
    client = Client(api_key=api_key)
    census = open_soma()
    
    # Load index
    celltype_list, embeddings_array, index = load_celltype_index(index_dir, verbose)
    
    if verbose:
        print("\n" + "="*70)
        print(f"RETRIEVING CELLS FOR: {cell_type_query}")
        print("="*70)
    
    # ========================================================================
    # STEP 1: Semantic search for matching cell type
    # ========================================================================
    
    if verbose:
        print("\n🔍 STEP 1: Finding best matching cell type...")
        print(f"   Query: '{cell_type_query}'")
    
    # Embed the query
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[cell_type_query]
    )
    
    # Extract embedding from result
    if hasattr(result, 'embeddings') and result.embeddings:
        query_embedding = np.array([result.embeddings[0].values], dtype='float32')
    elif isinstance(result, dict) and 'embeddings' in result:
        query_embedding = np.array([result['embeddings'][0]['values']], dtype='float32')
    else:
        raise ValueError("Failed to embed query")
    
    # Search hnswlib index for top 5 matches
    # Set ef parameter for search quality (higher = more accurate but slower)
    index.set_ef(200)
    labels, distances = index.knn_query(query_embedding, k=5)
    
    if verbose:
        print(f"   Top 5 matches:")
        for i, (label, dist) in enumerate(zip(labels[0], distances[0])):
            ct = celltype_list[int(label)]
            print(f"     {i+1}. {ct} (distance: {dist:.3f})")
    
    # Use the best match
    best_label = int(labels[0][0])
    target_cell_type = celltype_list[best_label]
    if verbose:
        print(f"\n✅ Using cell type: '{target_cell_type}'")
    
    # ========================================================================
    # STEP 2: Retrieve cells of target type
    # ========================================================================
    
    if verbose:
        print(f"\n📊 STEP 2: Retrieving {num_cells} cells of type '{target_cell_type}'...")
    
    obs_filter = f"cell_type == '{target_cell_type}' and dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc' and is_primary_data == True"
    
    obs_metadata = cellxgene_census.get_obs(
        census,
        organism="homo_sapiens",
        value_filter=obs_filter,
        column_names=["soma_joinid", "cell_type"]
    )
    
    soma_joinids_target = obs_metadata['soma_joinid'].tolist()
    if verbose:
        print(f"   Total {target_cell_type} cells available: {len(soma_joinids_target)}")
    
    # Sample if needed
    if len(soma_joinids_target) > num_cells:
        soma_joinids_target = list(np.random.choice(soma_joinids_target, size=num_cells, replace=False))
        if verbose:
            print(f"   Sampled: {len(soma_joinids_target)} cells")
    
    # ========================================================================
    # STEP 3: Retrieve control cells (NOT of target type)
    # ========================================================================
    
    if verbose:
        print(f"\n📊 STEP 3: Retrieving {num_controls} control cells (NOT '{target_cell_type}')...")
    
    obs_filter_control = f"cell_type != '{target_cell_type}' and dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc' and is_primary_data == True"
    
    obs_metadata_control = cellxgene_census.get_obs(
        census,
        organism="homo_sapiens",
        value_filter=obs_filter_control,
        column_names=["soma_joinid", "cell_type"]
    )
    
    soma_joinids_control = obs_metadata_control['soma_joinid'].tolist()
    if verbose:
        print(f"   Total control cells available: {len(soma_joinids_control)}")
    
    # Sample if needed
    if len(soma_joinids_control) > num_controls:
        soma_joinids_control = list(np.random.choice(soma_joinids_control, size=num_controls, replace=False))
        if verbose:
            print(f"   Sampled: {len(soma_joinids_control)} control cells")
    
    # ========================================================================
    # STEP 4: Query expression data in geneformer format
    # ========================================================================
    
    if verbose:
        print(f"\n🔬 STEP 4: Querying expression data for cells...")
    
    # Target cells
    if verbose:
        print(f"   Querying {len(soma_joinids_target)} target cells...")
    soma_str_target = ",".join(str(int(s)) for s in soma_joinids_target)
    obs_filter_expr = f"soma_joinid in [{soma_str_target}] and dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'"
    
    adata_target = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=obs_filter_expr,
        obs_column_names=["soma_joinid", "cell_type"]
    )
    
    if verbose:
        print(f"   ✓ Retrieved: {adata_target.n_obs} cells × {adata_target.n_vars} genes")
    
    # Control cells
    if verbose:
        print(f"   Querying {len(soma_joinids_control)} control cells...")
    soma_str_control = ",".join(str(int(s)) for s in soma_joinids_control)
    obs_filter_expr_ctrl = f"soma_joinid in [{soma_str_control}] and dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'"
    
    adata_control = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=obs_filter_expr_ctrl,
        obs_column_names=["soma_joinid", "cell_type"]
    )
    
    if verbose:
        print(f"   ✓ Retrieved: {adata_control.n_obs} cells × {adata_control.n_vars} genes")
    
    # ========================================================================
    # STEP 5: Convert to geneformer format
    # ========================================================================
    
    if verbose:
        print(f"\n💾 STEP 5: Converting to geneformer format...")
    
    # Get gene names
    gene_names = adata_target.var['feature_name'].tolist()
    
    # Convert sparse matrix to dense if needed
    if hasattr(adata_target.X, 'toarray'):
        X_target = adata_target.X.toarray()
    else:
        X_target = np.asarray(adata_target.X)
    
    if hasattr(adata_control.X, 'toarray'):
        X_control = adata_control.X.toarray()
    else:
        X_control = np.asarray(adata_control.X)
    
    # Get cell type labels
    cell_types_target = adata_target.obs['cell_type'].tolist()
    cell_types_control = adata_control.obs['cell_type'].tolist()
    
    # Geneformer format: [cell_type, gene_name, expression_value]
    # Each row = one gene in one cell
    
    data_cells = []
    for cell_idx, cell_type in enumerate(cell_types_target):
        for gene_idx, gene_name in enumerate(gene_names):
            expr_value = X_target[cell_idx, gene_idx]
            if expr_value > 0:  # Only include expressed genes (sparse)
                data_cells.append({
                    'cell_type': cell_type,
                    'gene_name': gene_name,
                    'expression_value': float(expr_value)
                })
    
    data_control = []
    for cell_idx, cell_type in enumerate(cell_types_control):
        for gene_idx, gene_name in enumerate(gene_names):
            expr_value = X_control[cell_idx, gene_idx]
            if expr_value > 0:  # Only include expressed genes (sparse)
                data_control.append({
                    'cell_type': cell_type,
                    'gene_name': gene_name,
                    'expression_value': float(expr_value)
                })
    
    # Convert to DataFrames
    df_cells = pd.DataFrame(data_cells)
    df_control = pd.DataFrame(data_control)
    
    if verbose:
        print(f"   ✓ Target cells data: {len(df_cells):,} gene expressions ({len(df_cells)/len(cell_types_target):.0f} genes/cell avg)")
        print(f"   ✓ Control cells data: {len(df_control):,} gene expressions ({len(df_control)/len(cell_types_control):.0f} genes/cell avg)")
    
    # ========================================================================
    # STEP 6: Save to CSV
    # ========================================================================
    
    if verbose:
        print(f"\n💾 STEP 6: Saving to CSV files...")
    
    output_files = {}
    
    if create_splits:
        # Split data into train/test/validation
        if verbose:
            print(f"   Creating train/test/val splits...")
        
        # Get indices for splitting
        n_target = len(df_cells)
        n_control = len(df_control)
        
        # Shuffle indices
        train_idx_target = int(n_target * split_fractions['train'])
        test_idx_target = train_idx_target + int(n_target * split_fractions['test'])
        
        train_idx_control = int(n_control * split_fractions['train'])
        test_idx_control = train_idx_control + int(n_control * split_fractions['test'])
        
        # Split target cells
        df_train_cells = df_cells.iloc[:train_idx_target].copy()
        df_test_cells = df_cells.iloc[train_idx_target:test_idx_target].copy()
        df_val_cells = df_cells.iloc[test_idx_target:].copy()
        
        # Split control cells
        df_train_control = df_control.iloc[:train_idx_control].copy()
        df_test_control = df_control.iloc[train_idx_control:test_idx_control].copy()
        df_val_control = df_control.iloc[test_idx_control:].copy()
        
        # Save split files
        for split_name, df_target, df_ctrl in [
            ('train', df_train_cells, df_train_control),
            ('test', df_test_cells, df_test_control),
            ('val', df_val_cells, df_val_control)
        ]:
            # Save target cells
            cells_path = os.path.join(output_dir, f"{output_prefix}_{split_name}_cells.csv")
            df_target.to_csv(cells_path, index=False)
            output_files[f'{split_name}_cells'] = cells_path
            
            if verbose:
                print(f"   ✓ Saved: {cells_path}")
                print(f"     Rows: {len(df_target):,}")
            
            # Save control cells
            controls_path = os.path.join(output_dir, f"{output_prefix}_{split_name}_controls.csv")
            df_ctrl.to_csv(controls_path, index=False)
            output_files[f'{split_name}_controls'] = controls_path
            
            if verbose:
                print(f"   ✓ Saved: {controls_path}")
                print(f"     Rows: {len(df_ctrl):,}")
    
    else:
        # Save as single files (no splits)
        cells_path = os.path.join(output_dir, f"{output_prefix}_cells.csv")
        df_cells.to_csv(cells_path, index=False)
        output_files['cells'] = cells_path
        if verbose:
            print(f"   ✓ Saved: {cells_path}")
            print(f"     Rows: {len(df_cells):,}")
        
        controls_path = os.path.join(output_dir, f"{output_prefix}_controls.csv")
        df_control.to_csv(controls_path, index=False)
        output_files['controls'] = controls_path
        if verbose:
            print(f"   ✓ Saved: {controls_path}")
            print(f"     Rows: {len(df_control):,}")
    
    # ========================================================================
    # STEP 7: Save metadata
    # ========================================================================
    
    metadata = {
        "query": cell_type_query,
        "matched_cell_type": target_cell_type,
        "num_cells_target": len(soma_joinids_target),
        "num_cells_control": len(soma_joinids_control),
        "num_genes": len(gene_names),
        "create_splits": create_splits,
        "split_fractions": split_fractions if create_splits else None,
        "output_files": output_files
    }
    
    metadata_path = os.path.join(output_dir, f"{output_prefix}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"   ✓ Saved: {metadata_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("✅ RETRIEVAL COMPLETE")
        print("="*70)
        print(f"\nQuery: {cell_type_query}")
        print(f"Matched cell type: {target_cell_type}")
        print(f"Target cells: {len(soma_joinids_target)}")
        print(f"Control cells: {len(soma_joinids_control)}")
        print(f"Total genes: {len(gene_names)}")
        
        if create_splits:
            print(f"\nSplit fractions:")
            for split, frac in split_fractions.items():
                print(f"  {split}: {frac*100:.0f}%")
        
        print(f"\nOutput files ({len(output_files)} files):")
        for key, path in output_files.items():
            print(f"  - {path}")
        print(f"  - {metadata_path}")
        print("="*70)
    
    return metadata


def main():
    """Command-line interface for retrieving cells and controls."""
    
    parser = argparse.ArgumentParser(
        description="Query cell types and retrieve expressing cells + controls in geneformer format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python retrieve_cells_and_controls.py --query "hepatocyte"
  
  # Specify output directory and prefix
  python retrieve_cells_and_controls.py --query "T cell" \\
    --output-dir ./data --output-prefix tcell_analysis
  
  # Custom cell counts and API key
  python retrieve_cells_and_controls.py --query "B cell" \\
    --num-cells 2000 --num-controls 2000 --api-key YOUR_KEY
  
  # Create train/test/val splits (will query 3x num_cells)
  python retrieve_cells_and_controls.py --query "macrophage" \\
    --num-cells 500 --num-controls 500 --create-splits \\
    --train-frac 0.7 --test-frac 0.15 --val-frac 0.15
  
  # Custom index location
  python retrieve_cells_and_controls.py --query "hepatocyte" \\
    --index-dir /path/to/index --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language cell type query (e.g., 'hepatocyte', 'T cell', 'immune cells')"
    )
    
    parser.add_argument(
        "--num-cells",
        type=int,
        default=1000,
        help="Number of target cell type cells to retrieve (default: 1000)"
    )
    
    parser.add_argument(
        "--num-controls",
        type=int,
        default=1000,
        help="Number of control cells (not of target type) to retrieve (default: 1000)"
    )
    
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/test/val splits instead of single files"
    )
    
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Training split fraction (default: 0.7)"
    )
    
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Test split fraction (default: 0.15)"
    )
    
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./geneformer_data",
        help="Directory to save output files (default: ./geneformer_data)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: derived from query)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (default: GOOGLE_API_KEY env var)"
    )
    
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./celltype_index",
        help="Directory containing prebuilt celltype index (default: ./celltype_index)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages (default: True)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key provided")
        print("   Set GOOGLE_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Default output prefix if not specified
    output_prefix = args.output_prefix or args.query.replace(" ", "_").lower()
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Build split fractions dict
    split_fractions = {
        'train': args.train_frac,
        'test': args.test_frac,
        'val': args.val_frac
    }
    
    # Retrieve cells
    result = retrieve_cells_and_controls(
        cell_type_query=args.query,
        num_cells=args.num_cells,
        num_controls=args.num_controls,
        output_dir=args.output_dir,
        output_prefix=output_prefix,
        api_key=api_key,
        index_dir=args.index_dir,
        create_splits=args.create_splits,
        split_fractions=split_fractions,
        verbose=verbose
    )
    
    return result


if __name__ == "__main__":
    main()

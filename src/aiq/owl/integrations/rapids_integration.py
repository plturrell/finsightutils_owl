"""
RAPIDS integration module for accelerated data processing.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

# Import RAPIDS libraries conditionally
try:
    import cudf
    import cuml
    import cugraph
    import cuspatial
    import cupy as cp
    from cuml.neighbors import NearestNeighbors
    from cuml.cluster import DBSCAN, KMeans
    from cuml.manifold import TSNE, UMAP
    from cuml.decomposition import PCA, TruncatedSVD
    from cugraph.community import louvain, leiden, ecg
    from cuml.metrics import pairwise_distances
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAPIDSIntegration:
    """
    RAPIDS integration for accelerated data processing and analytics.
    Provides GPU-accelerated data transformations, machine learning, and graph analytics.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        device_id: int = 0,
        memory_limit: Optional[int] = None,
        enable_diagnostics: bool = False,
    ):
        """
        Initialize the RAPIDS integration.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            device_id: GPU device ID to use
            memory_limit: Memory limit in bytes (None for no limit)
            enable_diagnostics: Enable performance diagnostics
        """
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.device_id = device_id
        self.memory_limit = memory_limit
        self.enable_diagnostics = enable_diagnostics
        
        # Performance tracking
        self.performance_logs = []
        
        # Initialize GPU if available
        if self.use_gpu:
            try:
                # Set CUDA device
                cp.cuda.Device(device_id).use()
                logger.info(f"RAPIDS integration using CUDA device {device_id}")
                
                # Test RAPIDS functionality
                test_df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
                test_result = test_df.sum().to_pandas()
                logger.info(f"RAPIDS test successful: {test_result}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize RAPIDS: {e}")
                self.use_gpu = False
        
        logger.info(
            f"RAPIDS integration initialized (use_gpu={self.use_gpu}, "
            f"device_id={self.device_id})"
        )
    
    def is_available(self) -> bool:
        """Check if RAPIDS is available and initialized."""
        return self.use_gpu and RAPIDS_AVAILABLE
    
    def _time_operation(self, operation_name: str):
        """Decorator for timing operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enable_diagnostics:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Log performance
                    self.performance_logs.append({
                        "operation": operation_name,
                        "duration": end_time - start_time,
                        "success": True,
                        "timestamp": time.time(),
                    })
                    
                    return result
                except Exception as e:
                    end_time = time.time()
                    
                    # Log error
                    self.performance_logs.append({
                        "operation": operation_name,
                        "duration": end_time - start_time,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time(),
                    })
                    
                    raise
            return wrapper
        return decorator
    
    def get_performance_logs(self) -> List[Dict[str, Any]]:
        """Get performance logs for operations."""
        return self.performance_logs
    
    @_time_operation("dataframe_to_gpu")
    def dataframe_to_gpu(self, df: pd.DataFrame) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Convert pandas DataFrame to cuDF DataFrame for GPU processing.
        
        Args:
            df: Pandas DataFrame to convert
            
        Returns:
            cuDF DataFrame if GPU available, otherwise the original pandas DataFrame
        """
        if not self.use_gpu:
            return df
        
        try:
            # Convert to cuDF
            gpu_df = cudf.DataFrame.from_pandas(df)
            logger.debug(f"Converted DataFrame to GPU (rows={len(gpu_df)})")
            return gpu_df
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to GPU: {e}")
            return df
    
    @_time_operation("dataframe_to_cpu")
    def dataframe_to_cpu(self, df: Union[cudf.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert cuDF DataFrame back to pandas DataFrame.
        
        Args:
            df: cuDF or pandas DataFrame
            
        Returns:
            pandas DataFrame
        """
        if not self.use_gpu or not isinstance(df, cudf.DataFrame):
            return df
        
        try:
            # Convert to pandas
            cpu_df = df.to_pandas()
            logger.debug(f"Converted DataFrame to CPU (rows={len(cpu_df)})")
            return cpu_df
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to CPU: {e}")
            raise
    
    @_time_operation("filter_dataframe")
    def filter_dataframe(
        self, 
        df: Union[cudf.DataFrame, pd.DataFrame],
        conditions: Dict[str, Any]
    ) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Filter DataFrame based on conditions.
        
        Args:
            df: DataFrame to filter
            conditions: Dictionary of column-value conditions
            
        Returns:
            Filtered DataFrame
        """
        # Convert to GPU if needed
        gpu_df = self.dataframe_to_gpu(df) if not isinstance(df, cudf.DataFrame) else df
        
        try:
            # Apply filters
            filtered_df = gpu_df
            
            for column, value in conditions.items():
                if column in gpu_df.columns:
                    if isinstance(value, (list, tuple)):
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
            
            logger.debug(
                f"Filtered DataFrame from {len(gpu_df)} to {len(filtered_df)} rows"
            )
            
            # Return in the same format as input
            if isinstance(df, pd.DataFrame):
                return self.dataframe_to_cpu(filtered_df)
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering DataFrame: {e}")
            
            # Fall back to pandas if error
            if isinstance(df, cudf.DataFrame):
                cpu_df = self.dataframe_to_cpu(df)
                
                # Apply filters in pandas
                filtered_df = cpu_df
                for column, value in conditions.items():
                    if column in cpu_df.columns:
                        if isinstance(value, (list, tuple)):
                            filtered_df = filtered_df[filtered_df[column].isin(value)]
                        else:
                            filtered_df = filtered_df[filtered_df[column] == value]
                
                return filtered_df
            
            # If already pandas, raise the error
            raise
    
    @_time_operation("join_dataframes")
    def join_dataframes(
        self,
        left_df: Union[cudf.DataFrame, pd.DataFrame],
        right_df: Union[cudf.DataFrame, pd.DataFrame],
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
    ) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Join two DataFrames with GPU acceleration.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            on: Column(s) to join on
            how: Join type (inner, left, right, outer)
            left_on: Left join column(s)
            right_on: Right join column(s)
            
        Returns:
            Joined DataFrame
        """
        # Determine output format
        output_pandas = isinstance(left_df, pd.DataFrame) or isinstance(right_df, pd.DataFrame)
        
        # Convert to GPU if needed
        left_gpu = self.dataframe_to_gpu(left_df) if not isinstance(left_df, cudf.DataFrame) else left_df
        right_gpu = self.dataframe_to_gpu(right_df) if not isinstance(right_df, cudf.DataFrame) else right_df
        
        try:
            # Perform join
            if on is not None:
                result = left_gpu.merge(right_gpu, on=on, how=how)
            else:
                result = left_gpu.merge(right_gpu, left_on=left_on, right_on=right_on, how=how)
            
            logger.debug(
                f"Joined DataFrames with {len(left_gpu)} and {len(right_gpu)} rows, "
                f"result has {len(result)} rows"
            )
            
            # Convert back to pandas if needed
            if output_pandas:
                return self.dataframe_to_cpu(result)
            return result
            
        except Exception as e:
            logger.error(f"Error joining DataFrames: {e}")
            
            # Fall back to pandas
            left_cpu = self.dataframe_to_cpu(left_gpu)
            right_cpu = self.dataframe_to_cpu(right_gpu)
            
            if on is not None:
                result = left_cpu.merge(right_cpu, on=on, how=how)
            else:
                result = left_cpu.merge(right_cpu, left_on=left_on, right_on=right_on, how=how)
            
            logger.warning(f"Performed join using pandas fallback")
            return result
    
    @_time_operation("group_by_agg")
    def group_by_agg(
        self,
        df: Union[cudf.DataFrame, pd.DataFrame],
        group_by: Union[str, List[str]],
        aggs: Dict[str, Union[str, List[str]]],
    ) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Group by and aggregate with GPU acceleration.
        
        Args:
            df: DataFrame to aggregate
            group_by: Column(s) to group by
            aggs: Dictionary mapping columns to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        # Determine output format
        output_pandas = isinstance(df, pd.DataFrame)
        
        # Convert to GPU if needed
        gpu_df = self.dataframe_to_gpu(df) if not isinstance(df, cudf.DataFrame) else df
        
        try:
            # Perform group by and aggregation
            grouped = gpu_df.groupby(group_by)
            result = grouped.agg(aggs)
            
            # Reset index
            result = result.reset_index()
            
            logger.debug(
                f"Grouped DataFrame with {len(gpu_df)} rows by {group_by}, "
                f"result has {len(result)} rows"
            )
            
            # Convert back to pandas if needed
            if output_pandas:
                return self.dataframe_to_cpu(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in group by aggregation: {e}")
            
            # Fall back to pandas
            cpu_df = self.dataframe_to_cpu(gpu_df)
            
            grouped = cpu_df.groupby(group_by)
            result = grouped.agg(aggs)
            result = result.reset_index()
            
            logger.warning(f"Performed group by using pandas fallback")
            return result
    
    @_time_operation("perform_pca")
    def perform_pca(
        self,
        df: Union[cudf.DataFrame, pd.DataFrame],
        n_components: int = 2,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[Union[cudf.DataFrame, pd.DataFrame], Any]:
        """
        Perform PCA dimensionality reduction.
        
        Args:
            df: DataFrame with features
            n_components: Number of principal components
            feature_columns: Columns to use as features (None for all)
            
        Returns:
            Tuple of (DataFrame with PCA results, PCA model)
        """
        if not self.use_gpu:
            # Fallback to sklearn
            from sklearn.decomposition import PCA as SklearnPCA
            
            # Select features
            if feature_columns is not None:
                features = df[feature_columns].values
            else:
                features = df.select_dtypes(include=['number']).values
            
            # Perform PCA
            pca = SklearnPCA(n_components=n_components)
            pca_result = pca.fit_transform(features)
            
            # Create result DataFrame
            result_df = pd.DataFrame(
                pca_result,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            
            # Add index from original DataFrame
            result_df.index = df.index
            
            logger.info(f"Performed PCA using scikit-learn (n_components={n_components})")
            return result_df, pca
        
        try:
            # Convert to GPU if needed
            gpu_df = self.dataframe_to_gpu(df) if not isinstance(df, cudf.DataFrame) else df
            
            # Select features
            if feature_columns is not None:
                features = gpu_df[feature_columns]
            else:
                features = gpu_df.select_dtypes(include=['number'])
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(features)
            
            # Create result DataFrame
            result_df = cudf.DataFrame(
                pca_result,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            
            # Add index from original DataFrame
            result_df.index = gpu_df.index
            
            logger.info(f"Performed PCA using RAPIDS (n_components={n_components})")
            
            # Convert back to pandas if needed
            if isinstance(df, pd.DataFrame):
                return self.dataframe_to_cpu(result_df), pca
            return result_df, pca
            
        except Exception as e:
            logger.error(f"Error performing PCA: {e}")
            
            # Fall back to sklearn
            from sklearn.decomposition import PCA as SklearnPCA
            
            # Convert to CPU
            cpu_df = self.dataframe_to_cpu(df) if isinstance(df, cudf.DataFrame) else df
            
            # Select features
            if feature_columns is not None:
                features = cpu_df[feature_columns].values
            else:
                features = cpu_df.select_dtypes(include=['number']).values
            
            # Perform PCA
            pca = SklearnPCA(n_components=n_components)
            pca_result = pca.fit_transform(features)
            
            # Create result DataFrame
            result_df = pd.DataFrame(
                pca_result,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            
            # Add index from original DataFrame
            result_df.index = cpu_df.index
            
            logger.warning(f"Performed PCA using scikit-learn fallback")
            return result_df, pca
    
    @_time_operation("perform_clustering")
    def perform_clustering(
        self,
        df: Union[cudf.DataFrame, pd.DataFrame],
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Union[cudf.DataFrame, pd.DataFrame], Any]:
        """
        Perform clustering on data.
        
        Args:
            df: DataFrame with features
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            feature_columns: Columns to use as features (None for all)
            **kwargs: Additional parameters for clustering algorithm
            
        Returns:
            Tuple of (DataFrame with cluster labels, Clustering model)
        """
        if not self.use_gpu:
            # Fallback to sklearn
            if method == "kmeans":
                from sklearn.cluster import KMeans as SklearnKMeans
                
                # Select features
                if feature_columns is not None:
                    features = df[feature_columns].values
                else:
                    features = df.select_dtypes(include=['number']).values
                
                # Auto-determine number of clusters if not specified
                if n_clusters is None:
                    # Use the square root heuristic
                    n_clusters = max(2, int(np.sqrt(len(df) / 2)))
                    logger.info(f"Auto-determined n_clusters = {n_clusters} for dataset with {len(df)} rows")
                
                # Perform clustering
                model = SklearnKMeans(n_clusters=n_clusters, **kwargs)
                labels = model.fit_predict(features)
                
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN as SklearnDBSCAN
                
                # Select features
                if feature_columns is not None:
                    features = df[feature_columns].values
                else:
                    features = df.select_dtypes(include=['number']).values
                
                # Auto-determine DBSCAN parameters if not provided
                eps = kwargs.get("eps")
                min_samples = kwargs.get("min_samples")
                
                if eps is None or min_samples is None:
                    # Auto-determine appropriate parameters based on data density
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Sample data for large datasets to speed up parameter estimation
                    sample_size = min(10000, len(features))
                    if len(features) > sample_size:
                        indices = np.random.choice(len(features), sample_size, replace=False)
                        sample_features = features[indices]
                    else:
                        sample_features = features
                    
                    # Compute distances to nearest neighbors
                    nn = NearestNeighbors(n_neighbors=2)
                    nn.fit(sample_features)
                    distances, _ = nn.kneighbors(sample_features)
                    
                    # Determine eps as the median of distances to nearest neighbor
                    if eps is None:
                        eps = np.median(distances[:, 1])
                        logger.info(f"Auto-determined eps = {eps:.4f} based on data density")
                    
                    # Determine min_samples based on dimensionality
                    if min_samples is None:
                        dimensionality = features.shape[1]
                        min_samples = max(5, dimensionality * 2)
                        logger.info(f"Auto-determined min_samples = {min_samples} based on data dimensionality")
                
                # Perform clustering
                model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(features)
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Create result DataFrame
            result_df = df.copy()
            result_df["cluster"] = labels
            
            logger.info(f"Performed {method} clustering using scikit-learn")
            return result_df, model
        
        try:
            # Convert to GPU if needed
            gpu_df = self.dataframe_to_gpu(df) if not isinstance(df, cudf.DataFrame) else df
            
            # Select features
            if feature_columns is not None:
                features = gpu_df[feature_columns]
            else:
                features = gpu_df.select_dtypes(include=['number'])
            
            # Perform clustering
            if method == "kmeans":
                # Auto-determine number of clusters if not specified
                if n_clusters is None:
                    # Use the square root heuristic with GPU optimization
                    # For larger datasets, we can afford more clusters with GPU acceleration
                    rows = len(gpu_df)
                    if rows < 10000:
                        n_clusters = max(2, int(np.sqrt(rows / 2)))
                    else:
                        # Scale more efficiently for large datasets
                        n_clusters = max(2, min(100, int(np.sqrt(rows / 2) * 1.5)))
                    
                    logger.info(f"Auto-determined n_clusters = {n_clusters} for dataset with {rows} rows (GPU)")
                
                # Auto-tune algorithm parameters based on available GPU memory
                if "max_iter" not in kwargs:
                    # Get free memory for auto-tuning
                    try:
                        free_memory = cp.cuda.Device().mem_info[0]
                        data_size = features.shape[0] * features.shape[1] * 4  # Approximate bytes
                        memory_ratio = free_memory / (data_size * 10)  # Safety factor
                        
                        # Scale iterations based on available memory
                        max_iter = min(300, max(50, int(100 * memory_ratio)))
                        kwargs["max_iter"] = max_iter
                        logger.info(f"Auto-tuned max_iter = {max_iter} based on GPU memory")
                    except Exception as e:
                        logger.warning(f"Could not auto-tune KMeans parameters: {e}")
                        kwargs["max_iter"] = 100  # Default
                
                model = KMeans(n_clusters=n_clusters, **kwargs)
                model.fit(features)
                labels = model.labels_
                
            elif method == "dbscan":
                # Auto-determine DBSCAN parameters if not provided
                eps = kwargs.get("eps")
                min_samples = kwargs.get("min_samples")
                
                if eps is None or min_samples is None:
                    # Auto-determine appropriate parameters based on data density
                    # This is more efficient on GPU for large datasets
                    from cuml.neighbors import NearestNeighbors
                    
                    # Sample data for very large datasets
                    sample_size = min(50000, len(features))
                    if len(features) > sample_size:
                        # GPU-optimized sampling
                        sample_indices = cp.random.choice(cp.arange(len(features)), size=sample_size, replace=False)
                        sample_features = features.iloc[sample_indices.get()]
                    else:
                        sample_features = features
                    
                    # Compute distances to nearest neighbors (GPU-accelerated)
                    nn = NearestNeighbors(n_neighbors=2)
                    nn.fit(sample_features)
                    distances, _ = nn.kneighbors(sample_features)
                    
                    # Determine eps as the median of distances to nearest neighbor
                    if eps is None:
                        # Convert to CPU for median calculation if needed
                        if hasattr(distances, 'to_pandas'):
                            distances_cpu = distances.to_pandas().values
                        else:
                            distances_cpu = distances.get()
                        
                        eps = float(np.median(distances_cpu[:, 1]))
                        logger.info(f"Auto-determined eps = {eps:.4f} based on data density (GPU)")
                    
                    # Determine min_samples based on dimensionality and dataset size
                    if min_samples is None:
                        dimensionality = features.shape[1]
                        # Scale min_samples with dataset size for GPU optimization
                        data_scale_factor = max(1, min(3, np.log10(len(features)) / 3))
                        min_samples = max(5, int(dimensionality * 2 * data_scale_factor))
                        logger.info(f"Auto-determined min_samples = {min_samples} based on data characteristics (GPU)")
                
                # Create model with optimized parameters
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(features)
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Create result DataFrame
            result_df = gpu_df.copy()
            result_df["cluster"] = labels
            
            logger.info(f"Performed {method} clustering using RAPIDS")
            
            # Convert back to pandas if needed
            if isinstance(df, pd.DataFrame):
                return self.dataframe_to_cpu(result_df), model
            return result_df, model
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            
            # Fall back to sklearn
            if method == "kmeans":
                from sklearn.cluster import KMeans as SklearnKMeans
                
                # Convert to CPU
                cpu_df = self.dataframe_to_cpu(df) if isinstance(df, cudf.DataFrame) else df
                
                # Select features
                if feature_columns is not None:
                    features = cpu_df[feature_columns].values
                else:
                    features = cpu_df.select_dtypes(include=['number']).values
                
                # Perform clustering
                model = SklearnKMeans(n_clusters=n_clusters, **kwargs)
                labels = model.fit_predict(features)
                
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN as SklearnDBSCAN
                
                # Convert to CPU
                cpu_df = self.dataframe_to_cpu(df) if isinstance(df, cudf.DataFrame) else df
                
                # Select features
                if feature_columns is not None:
                    features = cpu_df[feature_columns].values
                else:
                    features = cpu_df.select_dtypes(include=['number']).values
                
                # Get DBSCAN parameters
                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 5)
                
                # Perform clustering
                model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(features)
            
            # Create result DataFrame
            result_df = cpu_df.copy()
            result_df["cluster"] = labels
            
            logger.warning(f"Performed {method} clustering using scikit-learn fallback")
            return result_df, model
    
    @_time_operation("compute_graph_metrics")
    def compute_graph_metrics(
        self,
        edge_df: Union[cudf.DataFrame, pd.DataFrame],
        source_col: str = "source",
        target_col: str = "target",
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute graph metrics using RAPIDS.
        
        Args:
            edge_df: DataFrame with edge information
            source_col: Column name for source vertices
            target_col: Column name for target vertices
            weight_col: Column name for edge weights (None for unweighted)
            
        Returns:
            Dictionary of graph metrics
        """
        if not self.use_gpu:
            # Fallback to NetworkX
            import networkx as nx
            
            # Create NetworkX graph
            if weight_col is not None:
                G = nx.from_pandas_edgelist(
                    edge_df, 
                    source=source_col, 
                    target=target_col,
                    edge_attr=weight_col
                )
            else:
                G = nx.from_pandas_edgelist(
                    edge_df, 
                    source=source_col, 
                    target=target_col
                )
            
            # Compute basic metrics
            metrics = {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G) if nx.is_connected(G) else False,
                "connected_components": nx.number_connected_components(G),
            }
            
            # Compute additional metrics for small-to-medium graphs
            if G.number_of_nodes() < 10000:
                # Sample a subset of nodes for expensive metrics
                sample_size = min(100, G.number_of_nodes())
                sampled_nodes = list(G.nodes())[:sample_size]
                
                # Average clustering coefficient (sampled)
                metrics["avg_clustering"] = nx.average_clustering(G, nodes=sampled_nodes)
                
                # Average shortest path length (sampled)
                if nx.is_connected(G):
                    path_lengths = []
                    for i, node_i in enumerate(sampled_nodes):
                        for node_j in sampled_nodes[i+1:]:
                            try:
                                path_lengths.append(nx.shortest_path_length(G, node_i, node_j))
                            except nx.NetworkXNoPath:
                                pass
                    
                    if path_lengths:
                        metrics["avg_path_length"] = sum(path_lengths) / len(path_lengths)
                
                # Degree statistics
                degrees = [d for _, d in G.degree()]
                metrics["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
                metrics["max_degree"] = max(degrees) if degrees else 0
                metrics["min_degree"] = min(degrees) if degrees else 0
            
            logger.info(f"Computed graph metrics using NetworkX")
            return metrics
        
        try:
            # Convert to GPU if needed
            gpu_df = self.dataframe_to_gpu(edge_df) if not isinstance(edge_df, cudf.DataFrame) else edge_df
            
            # Create cuGraph graph
            if weight_col is not None:
                G = cugraph.Graph()
                G.from_cudf_edgelist(
                    gpu_df, 
                    source=source_col, 
                    destination=target_col,
                    edge_attr=weight_col
                )
            else:
                G = cugraph.Graph()
                G.from_cudf_edgelist(
                    gpu_df, 
                    source=source_col, 
                    destination=target_col
                )
            
            # Compute basic metrics
            metrics = {
                "num_nodes": G.number_of_vertices(),
                "num_edges": G.number_of_edges(),
            }
            
            # Compute graph density
            num_vertices = G.number_of_vertices()
            num_edges = G.number_of_edges()
            if num_vertices > 1:
                metrics["density"] = num_edges / (num_vertices * (num_vertices - 1))
            else:
                metrics["density"] = 0
            
            # Compute connected components
            cc = cugraph.components.connectivity.connected_components(G)
            unique_components = cc['labels'].nunique()
            metrics["connected_components"] = unique_components
            metrics["is_connected"] = unique_components == 1
            
            # Compute degrees
            degrees = G.degrees()
            metrics["avg_degree"] = degrees["degree"].mean()
            metrics["max_degree"] = degrees["degree"].max()
            metrics["min_degree"] = degrees["degree"].min()
            
            # For small-to-medium graphs, compute additional metrics
            if G.number_of_vertices() < 10000:
                # PageRank
                pagerank = cugraph.pagerank(G)
                metrics["pagerank_stats"] = {
                    "min": pagerank["pagerank"].min(),
                    "max": pagerank["pagerank"].max(),
                    "mean": pagerank["pagerank"].mean(),
                }
                
                # Community detection for reasonable sized graphs
                if G.number_of_vertices() < 5000:
                    try:
                        louvain_parts = louvain(G)
                        metrics["louvain_modularity"] = louvain_parts.modularity
                        metrics["louvain_communities"] = louvain_parts.partition.nunique()
                    except Exception as e:
                        logger.warning(f"Failed to compute Louvain: {e}")
            
            logger.info(f"Computed graph metrics using RAPIDS")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing graph metrics: {e}")
            
            # Fall back to NetworkX
            import networkx as nx
            
            # Convert to CPU
            cpu_df = self.dataframe_to_cpu(edge_df) if isinstance(edge_df, cudf.DataFrame) else edge_df
            
            # Create NetworkX graph
            if weight_col is not None:
                G = nx.from_pandas_edgelist(
                    cpu_df, 
                    source=source_col, 
                    target=target_col,
                    edge_attr=weight_col
                )
            else:
                G = nx.from_pandas_edgelist(
                    cpu_df, 
                    source=source_col, 
                    target=target_col
                )
            
            # Compute basic metrics
            metrics = {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G) if not G.is_directed() else None,
                "connected_components": nx.number_connected_components(G) if not G.is_directed() else None,
            }
            
            logger.warning(f"Computed graph metrics using NetworkX fallback")
            return metrics
    
    @_time_operation("find_shortest_paths")
    def find_shortest_paths(
        self,
        edge_df: Union[cudf.DataFrame, pd.DataFrame],
        source_vertices: List[Any],
        target_vertices: List[Any],
        source_col: str = "source",
        target_col: str = "target",
        weight_col: Optional[str] = None,
    ) -> Dict[Tuple[Any, Any], List[Any]]:
        """
        Find shortest paths between source and target vertices.
        
        Args:
            edge_df: DataFrame with edge information
            source_vertices: List of source vertices
            target_vertices: List of target vertices
            source_col: Column name for source vertices
            target_col: Column name for target vertices
            weight_col: Column name for edge weights (None for unweighted)
            
        Returns:
            Dictionary mapping (source, target) pairs to paths
        """
        if not self.use_gpu:
            # Fallback to NetworkX
            import networkx as nx
            
            # Create NetworkX graph
            if weight_col is not None:
                G = nx.from_pandas_edgelist(
                    edge_df, 
                    source=source_col, 
                    target=target_col,
                    edge_attr=weight_col
                )
            else:
                G = nx.from_pandas_edgelist(
                    edge_df, 
                    source=source_col, 
                    target=target_col
                )
            
            # Find shortest paths
            paths = {}
            for source in source_vertices:
                for target in target_vertices:
                    if source == target:
                        paths[(source, target)] = [source]
                        continue
                        
                    try:
                        if weight_col is not None:
                            path = nx.shortest_path(G, source, target, weight=weight_col)
                        else:
                            path = nx.shortest_path(G, source, target)
                        paths[(source, target)] = path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        paths[(source, target)] = []
            
            logger.info(f"Found shortest paths using NetworkX")
            return paths
        
        try:
            # Convert to GPU if needed
            gpu_df = self.dataframe_to_gpu(edge_df) if not isinstance(edge_df, cudf.DataFrame) else edge_df
            
            # Create cuGraph graph
            if weight_col is not None:
                G = cugraph.Graph()
                G.from_cudf_edgelist(
                    gpu_df, 
                    source=source_col, 
                    destination=target_col,
                    edge_attr=weight_col
                )
            else:
                G = cugraph.Graph()
                G.from_cudf_edgelist(
                    gpu_df, 
                    source=source_col, 
                    destination=target_col
                )
            
            # Find shortest paths
            paths = {}
            
            # Process each source vertex
            for source in source_vertices:
                # Run single-source shortest path from this source
                if weight_col is not None:
                    sssp = cugraph.sssp(G, source, weight_col)
                else:
                    sssp = cugraph.sssp(G, source)
                
                # Convert to pandas for easier processing
                sssp_df = sssp.to_pandas()
                
                # Extract paths for each target
                for target in target_vertices:
                    if source == target:
                        paths[(source, target)] = [source]
                        continue
                    
                    # Check if target is reachable
                    target_row = sssp_df[sssp_df["vertex"] == target]
                    if target_row.empty or target_row["distance"].iloc[0] == float("inf"):
                        paths[(source, target)] = []
                        continue
                    
                    # Reconstruct path from predecessor information
                    path = [target]
                    current = target
                    
                    while current != source:
                        predecessor = sssp_df[sssp_df["vertex"] == current]["predecessor"].iloc[0]
                        
                        # Check for cycles or invalid paths
                        if predecessor == -1 or predecessor in path:
                            paths[(source, target)] = []
                            break
                            
                        path.insert(0, predecessor)
                        current = predecessor
                    
                    if path and path[0] == source:
                        paths[(source, target)] = path
                    else:
                        paths[(source, target)] = []
            
            logger.info(f"Found shortest paths using RAPIDS")
            return paths
            
        except Exception as e:
            logger.error(f"Error finding shortest paths: {e}")
            
            # Fall back to NetworkX
            import networkx as nx
            
            # Convert to CPU
            cpu_df = self.dataframe_to_cpu(edge_df) if isinstance(edge_df, cudf.DataFrame) else edge_df
            
            # Create NetworkX graph
            if weight_col is not None:
                G = nx.from_pandas_edgelist(
                    cpu_df, 
                    source=source_col, 
                    target=target_col,
                    edge_attr=weight_col
                )
            else:
                G = nx.from_pandas_edgelist(
                    cpu_df, 
                    source=source_col, 
                    target=target_col
                )
            
            # Find shortest paths
            paths = {}
            for source in source_vertices:
                for target in target_vertices:
                    if source == target:
                        paths[(source, target)] = [source]
                        continue
                        
                    try:
                        if weight_col is not None:
                            path = nx.shortest_path(G, source, target, weight=weight_col)
                        else:
                            path = nx.shortest_path(G, source, target)
                        paths[(source, target)] = path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        paths[(source, target)] = []
            
            logger.warning(f"Found shortest paths using NetworkX fallback")
            return paths
    
    def cleanup(self) -> None:
        """Clean up GPU resources."""
        if self.use_gpu:
            try:
                # Clear CUDA caches
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                logger.info("Cleaned up CUDA memory pools")
            except Exception as e:
                logger.warning(f"Error cleaning up CUDA memory: {e}")
    
    def __del__(self):
        """Clean up when the object is garbage collected."""
        self.cleanup()
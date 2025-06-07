"""
NVIDIA Merlin integration for recommendation systems.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import uuid

# Import Merlin libraries conditionally
try:
    import cudf
    import nvtabular as nvt
    from nvtabular.ops import Categorify, Normalize, JoinGroupby, TargetEncoding
    from nvtabular.ops import LambdaOp, Operator, CategoryStatistics
    from nvtabular.workflow import Workflow
    
    import merlin.models.tf as mm
    from merlin.models.utils.dataset import unique_rows_by_features
    from merlin.schema.tags import Tags, TagsType
    
    MERLIN_AVAILABLE = True
except ImportError:
    MERLIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class MerlinIntegration:
    """
    NVIDIA Merlin integration for recommendation systems.
    Provides GPU-accelerated ETL, feature engineering, and model training.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Merlin integration.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            device_id: GPU device ID to use
            cache_dir: Directory for caching intermediate data
        """
        self.use_gpu = use_gpu and MERLIN_AVAILABLE
        self.device_id = device_id
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "merlin_cache")
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # Initialize state
        self.workflows = {}
        self.models = {}
        
        if self.use_gpu:
            try:
                # Test Merlin functionality
                import tensorflow as tf
                tf.config.experimental.set_visible_devices(
                    tf.config.list_physical_devices('GPU')[device_id], 'GPU'
                )
                logger.info(f"Merlin integration using CUDA device {device_id}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Merlin: {e}")
                self.use_gpu = False
        
        logger.info(
            f"Merlin integration initialized (use_gpu={self.use_gpu}, "
            f"device_id={self.device_id})"
        )
    
    def is_available(self) -> bool:
        """Check if Merlin is available and initialized."""
        return self.use_gpu and MERLIN_AVAILABLE
    
    def _ensure_cudf(self, df: Union[pd.DataFrame, cudf.DataFrame]) -> cudf.DataFrame:
        """
        Ensure that a DataFrame is a cuDF DataFrame.
        
        Args:
            df: Input DataFrame (pandas or cuDF)
            
        Returns:
            cuDF DataFrame
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if isinstance(df, cudf.DataFrame):
            return df
        return cudf.DataFrame.from_pandas(df)
    
    def _ensure_pandas(self, df: Union[pd.DataFrame, cudf.DataFrame]) -> pd.DataFrame:
        """
        Ensure that a DataFrame is a pandas DataFrame.
        
        Args:
            df: Input DataFrame (pandas or cuDF)
            
        Returns:
            pandas DataFrame
        """
        if isinstance(df, pd.DataFrame):
            return df
        return df.to_pandas()
    
    def create_workflow(
        self,
        workflow_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        """
        Create a new NVTabular workflow.
        
        Args:
            workflow_id: ID for the workflow (generated if None)
            cache_dir: Directory for caching workflow data
            
        Returns:
            Workflow ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        workflow_id = workflow_id or str(uuid.uuid4())
        
        # Set up cache directory
        workflow_cache_dir = cache_dir or os.path.join(self.cache_dir, f"workflow_{workflow_id}")
        os.makedirs(workflow_cache_dir, exist_ok=True)
        
        # Create empty workflow
        workflow = nvt.Workflow(client_files_path=workflow_cache_dir)
        
        # Store workflow
        self.workflows[workflow_id] = {
            "workflow": workflow,
            "cache_dir": workflow_cache_dir,
            "features": {},
            "created_at": time.time(),
        }
        
        logger.info(f"Created workflow {workflow_id}")
        return workflow_id
    
    def add_categorify(
        self,
        workflow_id: str,
        cat_columns: List[str],
        freq_threshold: int = 0,
        workflow_name: str = "categorify",
    ) -> str:
        """
        Add Categorify operation to a workflow.
        
        Args:
            workflow_id: Workflow ID
            cat_columns: List of categorical columns
            freq_threshold: Minimum frequency threshold
            workflow_name: Name for this part of the workflow
            
        Returns:
            Workflow ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Create Categorify operation
        cat_features = Categorify(
            cat_columns,
            freq_threshold=freq_threshold,
            name=workflow_name,
        )
        
        # Add to workflow features
        workflow_info["features"][workflow_name] = {
            "type": "categorify",
            "columns": cat_columns,
            "parameters": {"freq_threshold": freq_threshold},
        }
        
        logger.info(
            f"Added Categorify operation to workflow {workflow_id} "
            f"for columns {cat_columns}"
        )
        return workflow_id
    
    def add_normalization(
        self,
        workflow_id: str,
        num_columns: List[str],
        method: str = "standard",
        workflow_name: str = "normalize",
    ) -> str:
        """
        Add Normalize operation to a workflow.
        
        Args:
            workflow_id: Workflow ID
            num_columns: List of numerical columns
            method: Normalization method ('standard', 'minmax', 'log')
            workflow_name: Name for this part of the workflow
            
        Returns:
            Workflow ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Create Normalize operation
        norm_features = Normalize(
            num_columns,
            method=method,
            name=workflow_name,
        )
        
        # Add to workflow features
        workflow_info["features"][workflow_name] = {
            "type": "normalize",
            "columns": num_columns,
            "parameters": {"method": method},
        }
        
        logger.info(
            f"Added Normalize operation to workflow {workflow_id} "
            f"for columns {num_columns} using method {method}"
        )
        return workflow_id
    
    def add_target_encoding(
        self,
        workflow_id: str,
        cat_columns: List[str],
        target_columns: List[str],
        kfold: int = 5,
        fold_seed: int = 42,
        workflow_name: str = "target_encoding",
    ) -> str:
        """
        Add Target Encoding operation to a workflow.
        
        Args:
            workflow_id: Workflow ID
            cat_columns: List of categorical columns
            target_columns: List of target columns
            kfold: Number of folds for cross-validation
            fold_seed: Random seed for fold splitting
            workflow_name: Name for this part of the workflow
            
        Returns:
            Workflow ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Create Target Encoding operation
        target_encoding = TargetEncoding(
            cat_columns,
            target_columns,
            kfold=kfold,
            fold_seed=fold_seed,
            name=workflow_name,
        )
        
        # Add to workflow features
        workflow_info["features"][workflow_name] = {
            "type": "target_encoding",
            "columns": cat_columns,
            "parameters": {
                "target_columns": target_columns,
                "kfold": kfold,
                "fold_seed": fold_seed,
            },
        }
        
        logger.info(
            f"Added Target Encoding operation to workflow {workflow_id} "
            f"for columns {cat_columns} with targets {target_columns}"
        )
        return workflow_id
    
    def add_groupby_statistics(
        self,
        workflow_id: str,
        groupby_columns: List[str],
        target_columns: List[str],
        stats: List[str] = ["mean", "std"],
        workflow_name: str = "groupby_stats",
    ) -> str:
        """
        Add JoinGroupby operation to a workflow.
        
        Args:
            workflow_id: Workflow ID
            groupby_columns: List of columns to group by
            target_columns: List of columns to compute statistics on
            stats: List of statistics to compute
            workflow_name: Name for this part of the workflow
            
        Returns:
            Workflow ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Create JoinGroupby operation
        stats_dict = {}
        for col in target_columns:
            stats_dict[col] = stats
        
        groupby_stats = JoinGroupby(
            groupby_columns,
            stats_dict,
            name=workflow_name,
        )
        
        # Add to workflow features
        workflow_info["features"][workflow_name] = {
            "type": "join_groupby",
            "columns": groupby_columns,
            "parameters": {
                "target_columns": target_columns,
                "stats": stats,
            },
        }
        
        logger.info(
            f"Added JoinGroupby operation to workflow {workflow_id} "
            f"for columns {groupby_columns} with targets {target_columns}"
        )
        return workflow_id
    
    def fit_workflow(
        self,
        workflow_id: str,
        train_data: Union[pd.DataFrame, cudf.DataFrame],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fit a workflow on training data.
        
        Args:
            workflow_id: Workflow ID
            train_data: Training data
            output_path: Path to save the workflow
            
        Returns:
            Workflow statistics
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Ensure input is cuDF DataFrame
        if isinstance(train_data, pd.DataFrame):
            train_data = cudf.DataFrame.from_pandas(train_data)
        
        # Get list of all feature operations
        feature_ops = []
        for name, feature_info in workflow_info["features"].items():
            if feature_info["type"] == "categorify":
                feature_ops.append(
                    Categorify(
                        feature_info["columns"],
                        freq_threshold=feature_info["parameters"].get("freq_threshold", 0),
                        name=name,
                    )
                )
            elif feature_info["type"] == "normalize":
                feature_ops.append(
                    Normalize(
                        feature_info["columns"],
                        method=feature_info["parameters"].get("method", "standard"),
                        name=name,
                    )
                )
            elif feature_info["type"] == "target_encoding":
                feature_ops.append(
                    TargetEncoding(
                        feature_info["columns"],
                        feature_info["parameters"]["target_columns"],
                        kfold=feature_info["parameters"].get("kfold", 5),
                        fold_seed=feature_info["parameters"].get("fold_seed", 42),
                        name=name,
                    )
                )
            elif feature_info["type"] == "join_groupby":
                stats_dict = {}
                for col in feature_info["parameters"]["target_columns"]:
                    stats_dict[col] = feature_info["parameters"]["stats"]
                
                feature_ops.append(
                    JoinGroupby(
                        feature_info["columns"],
                        stats_dict,
                        name=name,
                    )
                )
        
        # Apply all features to the workflow
        for op in feature_ops:
            workflow.add_feature(op)
        
        # Fit the workflow
        start_time = time.time()
        workflow.fit(train_data)
        end_time = time.time()
        
        # Save the workflow if output path is provided
        if output_path:
            workflow.save(output_path)
            logger.info(f"Saved workflow to {output_path}")
        
        # Collect statistics
        stats = {
            "workflow_id": workflow_id,
            "training_time": end_time - start_time,
            "num_features": len(feature_ops),
            "num_rows": len(train_data),
            "num_columns": len(train_data.columns),
        }
        
        logger.info(
            f"Fitted workflow {workflow_id} on {len(train_data)} rows "
            f"in {end_time - start_time:.2f} seconds"
        )
        return stats
    
    def transform_workflow(
        self,
        workflow_id: str,
        data: Union[pd.DataFrame, cudf.DataFrame],
        output_format: str = "cudf",
    ) -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Transform data using a fitted workflow.
        
        Args:
            workflow_id: Workflow ID
            data: Data to transform
            output_format: Output format ('cudf' or 'pandas')
            
        Returns:
            Transformed data
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_info = self.workflows[workflow_id]
        workflow = workflow_info["workflow"]
        
        # Ensure input is cuDF DataFrame
        if isinstance(data, pd.DataFrame):
            data = cudf.DataFrame.from_pandas(data)
        
        # Transform the data
        start_time = time.time()
        transformed_data = workflow.transform(data)
        end_time = time.time()
        
        logger.info(
            f"Transformed {len(data)} rows using workflow {workflow_id} "
            f"in {end_time - start_time:.2f} seconds"
        )
        
        # Convert to requested output format
        if output_format.lower() == "pandas":
            return transformed_data.to_pandas()
        return transformed_data
    
    def create_recommendation_model(
        self,
        model_id: Optional[str] = None,
        model_type: str = "dlrm",
        embedding_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout_rate: Optional[float] = None,
    ) -> str:
        """
        Create a recommendation model.
        
        Args:
            model_id: ID for the model (generated if None)
            model_type: Type of model ('dlrm', 'dcn', 'ncf')
            embedding_dim: Dimension of embeddings
            num_layers: Number of layers in MLPs
            dropout_rate: Dropout rate
            
        Returns:
            Model ID
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        model_id = model_id or str(uuid.uuid4())
        
        # Set up adaptive model configuration based on hardware capabilities
        # These will be further refined during training based on the dataset
        
        # Auto-detect optimal embedding_dim based on GPU memory
        if embedding_dim is None:
            try:
                if self.use_gpu:
                    # Get GPU memory info
                    import cupy as cp
                    free_memory = cp.cuda.Device().mem_info[0] / (1024**3)  # Convert to GB
                    
                    # Scale embedding size based on available memory
                    if free_memory > 20:  # High-end GPU
                        embedding_dim = 256
                    elif free_memory > 10:  # Mid-range GPU
                        embedding_dim = 128
                    elif free_memory > 5:  # Low-end GPU
                        embedding_dim = 64
                    else:  # Limited memory
                        embedding_dim = 32
                else:
                    # Conservative for CPU
                    embedding_dim = 32
                    
                logger.info(f"Auto-determined embedding_dim = {embedding_dim}")
            except Exception as e:
                logger.warning(f"Could not auto-determine embedding_dim: {e}, using default")
                embedding_dim = 64
        
        # Auto-determine network depth based on model type and hardware
        if num_layers is None:
            if model_type == "dcn":
                # DCN benefits from more cross layers on powerful hardware
                num_layers = 3 if self.use_gpu else 2
            elif model_type == "dlrm":
                # DLRM can use deeper networks on GPU
                num_layers = 3 if self.use_gpu else 2
            else:
                # Default for other models
                num_layers = 2
                
            logger.info(f"Auto-determined num_layers = {num_layers} for {model_type}")
        
        # Auto-tune dropout rate based on dataset size (will be adjusted during training)
        if dropout_rate is None:
            # Default value - will be refined based on dataset
            dropout_rate = 0.1
        
        # Store model configuration (actual model will be created during training)
        self.models[model_id] = {
            "model": None,
            "type": model_type,
            "config": {
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
                "dropout_rate": dropout_rate,
                "auto_tuned": {
                    "embedding_dim": embedding_dim is None,
                    "num_layers": num_layers is None,
                    "dropout_rate": dropout_rate is None,
                }
            },
            "created_at": time.time(),
            "is_trained": False,
        }
        
        logger.info(
            f"Created {model_type} model configuration with ID {model_id}"
        )
        return model_id
    
    def train_recommendation_model(
        self,
        model_id: str,
        train_data: Union[pd.DataFrame, cudf.DataFrame],
        features: Dict[str, List[str]],
        target_column: str,
        workflow_id: Optional[str] = None,
        val_data: Optional[Union[pd.DataFrame, cudf.DataFrame]] = None,
        batch_size: int = 0,  # 0 means auto-determine based on data and GPU memory
        epochs: int = 0,      # 0 means auto-determine based on dataset size
        learning_rate: float = 0.001,
        output_path: Optional[str] = None,
        auto_tune: bool = True,  # Whether to auto-tune model parameters
    ) -> Dict[str, Any]:
        """
        Train a recommendation model.
        
        Args:
            model_id: Model ID
            train_data: Training data
            features: Dictionary mapping feature types to column names
                      (e.g., {"categorical": [...], "continuous": [...]})
            target_column: Name of the target column
            workflow_id: Optional workflow ID for preprocessing
            val_data: Optional validation data
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            output_path: Path to save the model
            
        Returns:
            Training history and metrics
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        model_info = self.models[model_id]
        model_type = model_info["type"]
        config = model_info["config"]
        
        # Preprocess data with workflow if provided
        if workflow_id:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            logger.info(f"Preprocessing data with workflow {workflow_id}")
            train_data = self.transform_workflow(workflow_id, train_data)
            
            if val_data is not None:
                val_data = self.transform_workflow(workflow_id, val_data)
        
        # Ensure data is in cuDF format
        if isinstance(train_data, pd.DataFrame):
            train_data = cudf.DataFrame.from_pandas(train_data)
            
        if val_data is not None and isinstance(val_data, pd.DataFrame):
            val_data = cudf.DataFrame.from_pandas(val_data)
        
        # Create schema
        schema = self._create_schema(train_data, features, target_column)
        
        # Create train and validation datasets
        train_dataset = self._create_dataset(
            train_data, schema, batch_size=batch_size
        )
        
        val_dataset = None
        if val_data is not None:
            val_dataset = self._create_dataset(
                val_data, schema, batch_size=batch_size
            )
        
        # Refine model parameters based on dataset characteristics
        embedding_dim = config["embedding_dim"]
        num_layers = config["num_layers"]
        dropout_rate = config["dropout_rate"]
        
        # Auto-tune parameters based on dataset if requested
        auto_tuned = config.get("auto_tuned", {})
        
        # Calculate dataset stats for auto-tuning
        dataset_size = len(train_data)
        num_categorical_features = len(features.get("categorical", [])) + len(features.get("user_id", [])) + len(features.get("item_id", []))
        num_continuous_features = len(features.get("continuous", []))
        
        # Refine embedding dimension based on categorical feature cardinality
        if auto_tuned.get("embedding_dim", False):
            # Calculate average cardinality
            avg_cardinality = 0
            total_features = 0
            
            for feature_type in ["categorical", "user_id", "item_id"]:
                for col in features.get(feature_type, []):
                    if col in train_data.columns:
                        cardinality = train_data[col].nunique()
                        avg_cardinality += cardinality
                        total_features += 1
            
            if total_features > 0:
                avg_cardinality /= total_features
                
                # Scale embedding dimension with cardinality but cap it
                # Higher cardinality needs larger embeddings
                # Use heuristic: embedding_dim ≈ min(600, 6 * cardinality^0.25)
                new_embedding_dim = min(600, int(6 * (avg_cardinality ** 0.25)))
                
                # Round to nearest power of 2 for GPU optimization
                new_embedding_dim = 2 ** int(np.log2(new_embedding_dim) + 0.5)
                
                # Update if significantly different
                if abs(new_embedding_dim - embedding_dim) > embedding_dim * 0.2:
                    logger.info(f"Auto-tuned embedding_dim from {embedding_dim} to {new_embedding_dim} based on dataset")
                    embedding_dim = new_embedding_dim
        
        # Refine network depth based on dataset size
        if auto_tuned.get("num_layers", False):
            # For very large datasets, deeper networks can learn more complex patterns
            if dataset_size > 1000000:
                new_num_layers = 4
            elif dataset_size > 100000:
                new_num_layers = 3
            else:
                new_num_layers = 2
                
            if new_num_layers != num_layers:
                logger.info(f"Auto-tuned num_layers from {num_layers} to {new_num_layers} based on dataset size")
                num_layers = new_num_layers
        
        # Refine dropout rate based on dataset size
        if auto_tuned.get("dropout_rate", False):
            # Smaller datasets need more regularization to prevent overfitting
            if dataset_size < 10000:
                new_dropout_rate = 0.3
            elif dataset_size < 100000:
                new_dropout_rate = 0.2
            elif dataset_size < 1000000:
                new_dropout_rate = 0.1
            else:
                new_dropout_rate = 0.05
                
            if abs(new_dropout_rate - dropout_rate) > 0.05:
                logger.info(f"Auto-tuned dropout_rate from {dropout_rate} to {new_dropout_rate} based on dataset size")
                dropout_rate = new_dropout_rate
        
        # Update model configuration with refined parameters
        config["embedding_dim"] = embedding_dim
        config["num_layers"] = num_layers
        config["dropout_rate"] = dropout_rate
        
        # Determine layer sizes based on model type and dataset
        if model_type == "dlrm":
            # Create a bottleneck architecture for better generalization
            bottom_mlp_sizes = [embedding_dim * 2] * (num_layers - 1) + [embedding_dim]
            top_mlp_sizes = [embedding_dim * 2] * (num_layers - 1) + [embedding_dim]
            
            model = mm.DLRMModel(
                schema,
                embedding_dim=embedding_dim,
                bottom_block=mm.MLPBlock(bottom_mlp_sizes),
                top_block=mm.MLPBlock(top_mlp_sizes, dropout=dropout_rate),
            )
        elif model_type == "dcn":
            # DCN works well with wider networks
            mlp_sizes = [embedding_dim * 2] * (num_layers - 1) + [embedding_dim]
            
            model = mm.DCNModel(
                schema,
                embedding_dim=embedding_dim,
                mlp_blocks=[mm.MLPBlock(mlp_sizes, dropout=dropout_rate)],
                num_cross_layers=num_layers,
            )
        elif model_type == "ncf":
            # NCF architecture with dynamic layer sizing
            mlp_sizes = [embedding_dim * 2] * (num_layers - 1) + [embedding_dim]
            
            model = mm.NCFModel(
                schema,
                embedding_dim=embedding_dim,
                mlp_block=mm.MLPBlock(mlp_sizes, dropout=dropout_rate),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Auto-tune optimizer and learning rate based on dataset size
        optimizer = None
        auto_learning_rate = learning_rate
        
        # Determine optimal optimizer and learning rate for dataset
        if dataset_size > 1000000:
            # Large datasets benefit from adaptive learning rates
            if self.use_gpu:
                import tensorflow as tf
                # Larger datasets, use more advanced optimizers on GPU
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=auto_learning_rate,
                        decay_steps=10000,
                        decay_rate=0.9,
                        staircase=True
                    )
                )
            else:
                # For CPU, simpler optimizer to avoid memory issues
                optimizer = mm.optimizers.Adam(learning_rate=auto_learning_rate)
        elif dataset_size > 100000:
            # Medium datasets with constant learning rate
            optimizer = mm.optimizers.Adam(learning_rate=auto_learning_rate)
        else:
            # Small datasets need higher learning rates with more regularization
            auto_learning_rate = min(0.01, learning_rate * 2)
            optimizer = mm.optimizers.Adam(learning_rate=auto_learning_rate)
        
        logger.info(f"Using learning rate: {auto_learning_rate} for dataset size: {dataset_size}")
        
        # Choose metrics based on task type (default to binary classification)
        metrics = ["accuracy", mm.metrics.AUC()]
        
        # Auto-detect proper loss function based on target values
        loss = "binary_crossentropy"  # Default for binary classification
        
        # Check target distribution for regression vs classification
        if target_column in train_data.columns:
            unique_targets = train_data[target_column].nunique()
            if unique_targets <= 2:
                # Binary classification
                loss = "binary_crossentropy"
            elif unique_targets <= 10:
                # Multi-class classification
                loss = "categorical_crossentropy"
                metrics = ["accuracy", "top_k_categorical_accuracy"]
            else:
                # Likely regression
                loss = "mse"
                metrics = ["mae", "mse"]
                
            logger.info(f"Auto-detected task type with {unique_targets} unique targets, using loss: {loss}")
        
        # Compile model with optimized parameters
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        
        # Auto-tune batch size based on dataset size and GPU memory if not specified
        auto_batch_size = batch_size
        if auto_batch_size <= 0:
            # Calculate optimal batch size
            if self.use_gpu:
                try:
                    import cupy as cp
                    free_memory = cp.cuda.Device().mem_info[0]
                    
                    # Estimate memory per sample (rough heuristic)
                    estimated_memory_per_sample = 4 * embedding_dim * (num_categorical_features + 2)
                    
                    # Target using 80% of free memory
                    target_memory = free_memory * 0.8
                    optimal_batch_size = int(target_memory / estimated_memory_per_sample)
                    
                    # Clamp to reasonable values (powers of 2 work best on GPU)
                    optimal_batch_size = min(8192, max(32, 2 ** int(np.log2(optimal_batch_size))))
                    auto_batch_size = optimal_batch_size
                    
                    logger.info(f"Auto-determined batch size: {auto_batch_size} based on GPU memory")
                except Exception as e:
                    logger.warning(f"Could not auto-determine batch size from GPU memory: {e}")
                    # Fallback based on dataset size
                    if dataset_size > 1000000:
                        auto_batch_size = 2048
                    elif dataset_size > 100000:
                        auto_batch_size = 1024
                    else:
                        auto_batch_size = 512
            else:
                # CPU-based batch size determination
                if dataset_size > 1000000:
                    auto_batch_size = 1024
                elif dataset_size > 100000:
                    auto_batch_size = 512
                else:
                    auto_batch_size = 256
                    
            logger.info(f"Using batch size: {auto_batch_size} for dataset size: {dataset_size}")
            
            # Recreate datasets with new batch size if needed
            if auto_batch_size != batch_size:
                train_dataset = self._create_dataset(
                    train_data, schema, batch_size=auto_batch_size
                )
                
                if val_data is not None:
                    val_dataset = self._create_dataset(
                        val_data, schema, batch_size=auto_batch_size
                    )
        
        # Auto-determine optimal epochs based on dataset size
        auto_epochs = epochs
        if epochs <= 0:
            # Heuristic for epochs: smaller datasets need more epochs
            if dataset_size < 10000:
                auto_epochs = 50
            elif dataset_size < 100000:
                auto_epochs = 30
            elif dataset_size < 1000000:
                auto_epochs = 20
            else:
                auto_epochs = 10
                
            logger.info(f"Auto-determined epochs: {auto_epochs} based on dataset size: {dataset_size}")
        
        # Configure early stopping based on dataset size
        early_stopping = None
        try:
            import tensorflow as tf
            # Add early stopping for larger datasets
            if dataset_size > 50000:
                patience = max(3, int(auto_epochs / 10))
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_dataset else 'loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
                logger.info(f"Configured early stopping with patience: {patience}")
        except Exception as e:
            logger.warning(f"Could not configure early stopping: {e}")
        
        # Configure callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(early_stopping)
        
        # Train model with optimized parameters
        start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=auto_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        end_time = time.time()
        
        # Save model if output path is provided
        if output_path:
            model.save(output_path)
            logger.info(f"Saved model to {output_path}")
        
        # Update model info
        model_info["model"] = model
        model_info["schema"] = schema
        model_info["is_trained"] = True
        model_info["training_time"] = end_time - start_time
        
        # Collect metrics
        metrics = {
            "model_id": model_id,
            "training_time": end_time - start_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "final_loss": float(history.history["loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1]),
            "final_auc": float(history.history["auc"][-1]) if "auc" in history.history else None,
        }
        
        if val_dataset:
            metrics["final_val_loss"] = float(history.history["val_loss"][-1])
            metrics["final_val_accuracy"] = float(history.history["val_accuracy"][-1])
            metrics["final_val_auc"] = float(history.history["val_auc"][-1]) if "val_auc" in history.history else None
        
        logger.info(
            f"Trained {model_type} model {model_id} for {epochs} epochs "
            f"in {end_time - start_time:.2f} seconds"
        )
        return metrics
    
    def _create_schema(
        self,
        df: cudf.DataFrame,
        features: Dict[str, List[str]],
        target_column: str,
    ) -> mm.Schema:
        """
        Create a schema for the model.
        
        Args:
            df: DataFrame
            features: Dictionary mapping feature types to column names
            target_column: Name of the target column
            
        Returns:
            Merlin schema
        """
        # Create feature schema
        schema_dict = {}
        
        # Categorical features
        if "categorical" in features:
            for col in features["categorical"]:
                # Get unique values
                cardinality = df[col].nunique()
                schema_dict[col] = mm.schema.ColumnSchema(
                    tags=[Tags.CATEGORICAL],
                    properties={"domain": {"min": 0, "max": int(cardinality) - 1}},
                )
        
        # Continuous features
        if "continuous" in features:
            for col in features["continuous"]:
                schema_dict[col] = mm.schema.ColumnSchema(tags=[Tags.CONTINUOUS])
        
        # User ID features
        if "user_id" in features:
            for col in features["user_id"]:
                cardinality = df[col].nunique()
                schema_dict[col] = mm.schema.ColumnSchema(
                    tags=[Tags.CATEGORICAL, Tags.USER_ID],
                    properties={"domain": {"min": 0, "max": int(cardinality) - 1}},
                )
        
        # Item ID features
        if "item_id" in features:
            for col in features["item_id"]:
                cardinality = df[col].nunique()
                schema_dict[col] = mm.schema.ColumnSchema(
                    tags=[Tags.CATEGORICAL, Tags.ITEM_ID],
                    properties={"domain": {"min": 0, "max": int(cardinality) - 1}},
                )
        
        # Target column
        schema_dict[target_column] = mm.schema.ColumnSchema(tags=[Tags.TARGET])
        
        # Create schema
        schema = mm.Schema(schema_dict)
        return schema
    
    def _create_dataset(
        self,
        df: cudf.DataFrame,
        schema: mm.Schema,
        batch_size: int = 1024,
    ) -> mm.Dataset:
        """
        Create a dataset for model training.
        
        Args:
            df: DataFrame
            schema: Merlin schema
            batch_size: Batch size
            
        Returns:
            Merlin dataset
        """
        # Create dataset
        dataset = mm.Dataset(df, schema=schema)
        
        # Create data loader
        dataloader = mm.Dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        return dataloader
    
    def predict(
        self,
        model_id: str,
        data: Union[pd.DataFrame, cudf.DataFrame],
        workflow_id: Optional[str] = None,
        batch_size: int = 1024,
        output_format: str = "cudf",
    ) -> Union[pd.DataFrame, cudf.DataFrame, np.ndarray]:
        """
        Make predictions with a trained model.
        
        Args:
            model_id: Model ID
            data: Data to predict on
            workflow_id: Optional workflow ID for preprocessing
            batch_size: Batch size for prediction
            output_format: Output format ('cudf', 'pandas', or 'numpy')
            
        Returns:
            Predictions
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        model_info = self.models[model_id]
        
        if not model_info["is_trained"]:
            raise ValueError(f"Model {model_id} is not trained")
            
        model = model_info["model"]
        schema = model_info["schema"]
        
        # Preprocess data with workflow if provided
        if workflow_id:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            logger.info(f"Preprocessing data with workflow {workflow_id}")
            data = self.transform_workflow(workflow_id, data)
        
        # Ensure data is in cuDF format
        if isinstance(data, pd.DataFrame):
            data = cudf.DataFrame.from_pandas(data)
        
        # Create dataset
        dataset = mm.Dataset(data, schema=schema)
        
        # Create data loader
        dataloader = mm.Dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Make predictions
        start_time = time.time()
        predictions = model.predict(dataloader)
        end_time = time.time()
        
        logger.info(
            f"Made predictions on {len(data)} rows using model {model_id} "
            f"in {end_time - start_time:.2f} seconds"
        )
        
        # Return predictions in requested format
        if output_format.lower() == "pandas":
            return pd.DataFrame(predictions, columns=["prediction"])
        elif output_format.lower() == "cudf":
            return cudf.DataFrame(predictions, columns=["prediction"])
        return predictions
    
    def evaluate_model(
        self,
        model_id: str,
        data: Union[pd.DataFrame, cudf.DataFrame],
        target_column: str,
        workflow_id: Optional[str] = None,
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_id: Model ID
            data: Data to evaluate on
            target_column: Name of the target column
            workflow_id: Optional workflow ID for preprocessing
            batch_size: Batch size for evaluation
            
        Returns:
            Evaluation metrics
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration is not available")
            
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        model_info = self.models[model_id]
        
        if not model_info["is_trained"]:
            raise ValueError(f"Model {model_id} is not trained")
            
        model = model_info["model"]
        schema = model_info["schema"]
        
        # Preprocess data with workflow if provided
        if workflow_id:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            logger.info(f"Preprocessing data with workflow {workflow_id}")
            data = self.transform_workflow(workflow_id, data)
        
        # Ensure data is in cuDF format
        if isinstance(data, pd.DataFrame):
            data = cudf.DataFrame.from_pandas(data)
        
        # Create dataset
        dataset = mm.Dataset(data, schema=schema)
        
        # Create data loader
        dataloader = mm.Dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Evaluate model
        start_time = time.time()
        metrics = model.evaluate(dataloader)
        end_time = time.time()
        
        # Convert metrics to dictionary
        metrics_dict = {name: float(value) for name, value in zip(model.metrics_names, metrics)}
        metrics_dict["evaluation_time"] = end_time - start_time
        
        logger.info(
            f"Evaluated model {model_id} on {len(data)} rows "
            f"in {end_time - start_time:.2f} seconds"
        )
        return metrics_dict
    
    def cleanup(self) -> None:
        """Clean up resources."""
        import tensorflow as tf
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Clear GPU memory
        if self.use_gpu:
            try:
                # Clear CUDA caches
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                logger.info("Cleaned up CUDA memory pools")
            except Exception as e:
                logger.warning(f"Error cleaning up CUDA memory: {e}")
    
    def __del__(self):
        """Clean up when the object is garbage collected."""
        self.cleanup()
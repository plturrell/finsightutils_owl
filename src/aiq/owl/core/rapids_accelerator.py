"""
RAPIDS acceleration for knowledge graph construction and processing.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import logging
import os
import uuid
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

# Import RAPIDS libraries conditionally
try:
    import cudf
    import cugraph
    import cupy as cp
    from cugraph.experimental import PropertyGraph
    from cugraph import Graph as CuGraph
    import rmm  # RAPIDS Memory Manager
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    
# Import RDF libraries
import rdflib
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, XSD, OWL

logger = logging.getLogger(__name__)

class RAPIDSAccelerator:
    """
    Accelerates knowledge graph operations using NVIDIA RAPIDS.
    Provides GPU-accelerated graph processing for OWL/RDF data.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        memory_limit: Optional[int] = None,
        device_id: int = 0,
        pool_allocator: bool = True,
    ):
        """
        Initialize the RAPIDS accelerator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            memory_limit: GPU memory limit in bytes (None for no limit)
            device_id: GPU device ID to use
            pool_allocator: Whether to use RAPIDS Memory Manager pool allocator
        """
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.device_id = device_id
        self.memory_limit = memory_limit
        self.pool_allocator = pool_allocator
        
        # Track performance stats
        self._stats = {
            "available": self.use_gpu and RAPIDS_AVAILABLE,
            "device_id": device_id,
            "memory_limit": memory_limit,
            "operations": {
                "rdf_to_property_graph": {"count": 0, "total_time": 0, "avg_time": 0},
                "owl_generation": {"count": 0, "total_time": 0, "avg_time": 0},
                "find_related_entities": {"count": 0, "total_time": 0, "avg_time": 0},
            }
        }
        
        # Initialize GPU if available
        if self.use_gpu:
            try:
                # Initialize CUDA context
                cp.cuda.Device(device_id).use()
                logger.info("Using CUDA device %d", device_id)
                
                # Set up RMM pool allocator if requested
                if pool_allocator and RAPIDS_AVAILABLE:
                    try:
                        if memory_limit:
                            # Configure RMM with memory limit
                            rmm.reinitialize(
                                pool_allocator=True,
                                managed_memory=True,
                                initial_pool_size=memory_limit,
                                maximum_pool_size=memory_limit,
                                devices=device_id
                            )
                            logger.info(f"Initialized RMM pool allocator with memory limit {memory_limit / (1024**2):.2f} MB")
                        else:
                            # Use default memory limits
                            rmm.reinitialize(
                                pool_allocator=True, 
                                managed_memory=True,
                                devices=device_id
                            )
                            logger.info("Initialized RMM pool allocator with default memory limits")
                    except Exception as e:
                        logger.warning(f"Failed to initialize RMM pool allocator: {e}")
                
                # Test CUDA availability with a simple operation
                test_array = cp.array([1, 2, 3])
                result = cp.sum(test_array).get()
                logger.info(f"CUDA test successful: sum = {result}")
                
                # Get and log GPU info
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    logger.info(f"GPU info: {json.dumps(gpu_info)}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA: {e}")
                self.use_gpu = False
                self._stats["available"] = False
        
        logger.info(
            "RAPIDSAccelerator initialized with use_gpu=%s, RAPIDS_AVAILABLE=%s, device_id=%d",
            self.use_gpu,
            RAPIDS_AVAILABLE,
            self.device_id,
        )
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get information about the GPU device."""
        if not self.use_gpu:
            return {}
            
        try:
            device = cp.cuda.Device(self.device_id)
            attributes = device.attributes
            
            # Get memory info
            mem_info = device.mem_info
            total_memory = mem_info[0]
            free_memory = mem_info[1]
            
            return {
                "name": device.name,
                "compute_capability": f"{attributes.get('computeCapabilityMajor')}.{attributes.get('computeCapabilityMinor')}",
                "total_memory": total_memory,
                "free_memory": free_memory,
                "memory_used_percent": ((total_memory - free_memory) / total_memory) * 100,
                "multi_processor_count": attributes.get('multiProcessorCount', 0),
                "max_threads_per_block": attributes.get('maxThreadsPerBlock', 0),
                "max_shared_memory_per_block": attributes.get('maxSharedMemoryPerBlock', 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the accelerator.
        
        Returns:
            Dictionary of performance statistics
        """
        # Update with current GPU info
        if self.use_gpu:
            self._stats.update({"gpu_info": self._get_gpu_info()})
            
            # Update average times
            for op_name, stats in self._stats["operations"].items():
                if stats["count"] > 0:
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    
        return self._stats
    
    def rdf_to_property_graph(self, rdf_graph: Graph) -> Union[PropertyGraph, None]:
        """
        Convert an RDF graph to a RAPIDS PropertyGraph for efficient processing.
        
        Args:
            rdf_graph: RDF graph to convert
            
        Returns:
            RAPIDS PropertyGraph or None if RAPIDS is not available
        """
        if not self.use_gpu:
            logger.warning("RAPIDS acceleration not available, returning None")
            return None
        
        start_time = time.time()
        logger.info("Converting RDF graph with %d triples to PropertyGraph", len(rdf_graph))
        
        try:
            # Extract triples from RDF graph
            triples = []
            for s, p, o in rdf_graph:
                # Convert URIRefs, Literals, and BNodes to strings for RAPIDS
                s_str = str(s)
                p_str = str(p)
                
                # Handle different types of objects
                if isinstance(o, URIRef) or isinstance(o, BNode):
                    o_str = str(o)
                    o_type = "uri"
                elif isinstance(o, Literal):
                    o_str = str(o)
                    o_type = "literal"
                    # Extract datatype and language for enhanced processing
                    o_datatype = str(o.datatype) if o.datatype else ""
                    o_lang = o.language if o.language else ""
                else:
                    o_str = str(o)
                    o_type = "unknown"
                    o_datatype = ""
                    o_lang = ""
                
                triples.append((s_str, p_str, o_str, o_type, o_datatype, o_lang))
            
            # Create DataFrames for nodes and edges
            if RAPIDS_AVAILABLE:
                # Use cuDF for GPU acceleration
                edges_df = cudf.DataFrame(
                    triples, 
                    columns=["subject", "predicate", "object", "object_type", "datatype", "language"]
                )
                
                # Extract unique nodes
                subjects = edges_df["subject"].unique()
                objects = edges_df[edges_df["object_type"] == "uri"]["object"].unique()
                
                # Concatenate and get unique nodes
                all_nodes = cudf.concat([subjects, objects]).unique()
                
                # Create nodes DataFrame with node types
                nodes_df = cudf.DataFrame({"node_id": all_nodes})
                
                # Determine node types (could be subject, object, or both)
                is_subject = nodes_df["node_id"].isin(subjects)
                is_object = nodes_df["node_id"].isin(objects)
                
                # Create node types
                nodes_df["node_type"] = "unknown"
                nodes_df.loc[is_subject & is_object, "node_type"] = "both"
                nodes_df.loc[is_subject & ~is_object, "node_type"] = "subject_only"
                nodes_df.loc[~is_subject & is_object, "node_type"] = "object_only"
                
                # Create PropertyGraph
                property_graph = PropertyGraph()
                
                # Add nodes with their types
                property_graph.add_node_data(
                    nodes_df,
                    node_col_name="node_id",
                    type_name="node"
                )
                
                # Add edges with their attributes
                property_graph.add_edge_data(
                    edges_df,
                    source="subject",
                    destination="object",
                    edge_attr_cols=["predicate", "object_type", "datatype", "language"],
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Update stats
                self._stats["operations"]["rdf_to_property_graph"]["count"] += 1
                self._stats["operations"]["rdf_to_property_graph"]["total_time"] += elapsed_time
                
                logger.info(
                    "Created PropertyGraph with %d nodes and %d edges in %.2f seconds",
                    len(nodes_df),
                    len(edges_df),
                    elapsed_time
                )
                
                return property_graph
                
            else:
                # Fallback to pandas for CPU processing
                edges_df = pd.DataFrame(
                    triples, 
                    columns=["subject", "predicate", "object", "object_type", "datatype", "language"]
                )
                logger.warning(
                    "RAPIDS not available, created pandas DataFrame with %d edges",
                    len(edges_df),
                )
                return None
                
        except Exception as e:
            logger.error(
                "Error converting RDF graph to PropertyGraph: %s", str(e), exc_info=True
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update stats even for failed operations
            self._stats["operations"]["rdf_to_property_graph"]["count"] += 1
            self._stats["operations"]["rdf_to_property_graph"]["total_time"] += elapsed_time
            
            return None
    
    def accelerate_owl_generation(
        self, triples: List[Tuple[str, str, str]], base_uri: str
    ) -> Graph:
        """
        Accelerate the generation of OWL triples using RAPIDS.
        
        Args:
            triples: List of (subject, predicate, object) triples as strings
            base_uri: Base URI for the generated resources
            
        Returns:
            RDF graph with the generated triples
        """
        start_time = time.time()
        
        if not self.use_gpu:
            # Fallback to standard RDFlib processing
            logger.warning("RAPIDS acceleration not available, using standard processing")
            graph = Graph()
            
            for s, p, o in triples:
                s_uri = URIRef(s) if s.startswith("http") else URIRef(f"{base_uri}{s}")
                p_uri = URIRef(p) if p.startswith("http") else URIRef(f"{base_uri}{p}")
                
                # Handle different object types
                if o.startswith("http") or o.startswith("_:"):
                    o_node = URIRef(o)
                else:
                    # Simple string literal for now
                    o_node = Literal(o)
                
                graph.add((s_uri, p_uri, o_node))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"Generated RDF graph with {len(graph)} triples using CPU in {elapsed_time:.2f} seconds")
            return graph
        
        try:
            logger.info("Accelerating OWL generation with RAPIDS for %d triples", len(triples))
            
            # Create batches of triples for processing
            # This improves memory management on the GPU
            batch_size = 10000
            all_batches = [triples[i:i + batch_size] for i in range(0, len(triples), batch_size)]
            logger.info(f"Processing {len(all_batches)} batches of {batch_size} triples")
            
            # Final RDF graph to return
            final_graph = Graph()
            
            for batch_idx, batch in enumerate(all_batches):
                logger.debug(f"Processing batch {batch_idx+1}/{len(all_batches)}")
                
                # Process each batch on GPU
                # Convert triples to cuDF DataFrame
                if not batch:
                    continue
                    
                # Process object types in batch
                processed_batch = []
                for s, p, o in batch:
                    # Process subject and predicate
                    s_processed = s if s.startswith("http") else f"{base_uri}{s}"
                    p_processed = p if p.startswith("http") else f"{base_uri}{p}"
                    
                    # Process object - detect if it's a URI or literal
                    if o.startswith("http") or o.startswith("_:"):
                        o_processed = o
                        o_type = "uri"
                    else:
                        o_processed = o
                        o_type = "literal"
                        
                    processed_batch.append((s_processed, p_processed, o_processed, o_type))
                
                # Convert to cuDF DataFrame for GPU processing
                try:
                    batch_df = cudf.DataFrame(
                        processed_batch,
                        columns=["subject", "predicate", "object", "object_type"]
                    )
                    
                    # Process on GPU
                    # Additional processing could be done here like:
                    # - Pattern matching
                    # - Graph analytics
                    # - Inference
                    
                    # Convert back to CPU for RDFlib
                    cpu_df = batch_df.to_pandas()
                except Exception as e:
                    logger.warning(f"Error in cuDF processing: {e}, falling back to pandas")
                    # Fallback to pandas if cuDF processing fails
                    cpu_df = pd.DataFrame(
                        processed_batch,
                        columns=["subject", "predicate", "object", "object_type"]
                    )
                
                # Add triples to the RDF graph
                for _, row in cpu_df.iterrows():
                    s = URIRef(row["subject"])
                    p = URIRef(row["predicate"])
                    
                    # Handle different object types
                    if row["object_type"] == "uri":
                        o = URIRef(row["object"])
                    else:
                        # Extract datatype from literal if possible
                        obj = row["object"]
                        
                        # Try to infer datatype for common literals
                        try:
                            # Check if it's an integer
                            int(obj)
                            o = Literal(obj, datatype=XSD.integer)
                        except ValueError:
                            try:
                                # Check if it's a float
                                float(obj)
                                o = Literal(obj, datatype=XSD.decimal)
                            except ValueError:
                                # Check if it's a boolean
                                if obj.lower() in ("true", "false"):
                                    o = Literal(obj.lower() == "true", datatype=XSD.boolean)
                                else:
                                    # Default to string literal
                                    o = Literal(obj)
                    
                    final_graph.add((s, p, o))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update stats
            self._stats["operations"]["owl_generation"]["count"] += 1
            self._stats["operations"]["owl_generation"]["total_time"] += elapsed_time
            
            logger.info("Generated RDF graph with %d triples in %.2f seconds using RAPIDS", len(final_graph), elapsed_time)
            return final_graph
            
        except Exception as e:
            logger.error(
                "Error in accelerated OWL generation: %s", str(e), exc_info=True
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update stats even for failed operations
            self._stats["operations"]["owl_generation"]["count"] += 1
            self._stats["operations"]["owl_generation"]["total_time"] += elapsed_time
            
            # Fallback to standard processing
            logger.warning("Falling back to CPU-based RDF processing")
            return self._fallback_owl_generation(triples, base_uri)
    
    def _fallback_owl_generation(self, triples: List[Tuple[str, str, str]], base_uri: str) -> Graph:
        """
        Fallback CPU implementation for OWL generation when GPU processing fails.
        
        Args:
            triples: List of (subject, predicate, object) triples as strings
            base_uri: Base URI for the generated resources
            
        Returns:
            RDF graph with the generated triples
        """
        start_time = time.time()
        graph = Graph()
        
        for s, p, o in triples:
            s_uri = URIRef(s) if s.startswith("http") else URIRef(f"{base_uri}{s}")
            p_uri = URIRef(p) if p.startswith("http") else URIRef(f"{base_uri}{p}")
            
            # Handle different object types
            if o.startswith("http") or o.startswith("_:"):
                o_node = URIRef(o)
            else:
                # Simple string literal for now
                o_node = Literal(o)
            
            graph.add((s_uri, p_uri, o_node))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"Generated RDF graph with {len(graph)} triples using CPU fallback in {elapsed_time:.2f} seconds")
        return graph
    
    def find_related_entities(
        self, property_graph: PropertyGraph, entity_uri: str, max_distance: int = 2
    ) -> Dict[str, Any]:
        """
        Find entities related to the given entity within a maximum distance.
        
        Args:
            property_graph: RAPIDS PropertyGraph
            entity_uri: URI of the entity to find related entities for
            max_distance: Maximum distance (number of hops) to search
            
        Returns:
            Dictionary of related entities grouped by relationship type
        """
        start_time = time.time()
        
        if not self.use_gpu or property_graph is None:
            logger.warning("RAPIDS acceleration not available for entity relation finding")
            return {"related_entities": []}
        
        try:
            logger.info(
                "Finding entities related to %s within %d hops", entity_uri, max_distance
            )
            
            # Check if entity exists in the graph
            # First get all nodes from the property graph
            all_nodes = property_graph.nodes
            node_ids = all_nodes["node_id"].to_pandas() if "node_id" in all_nodes.columns else []
            
            # Check if entity_uri is in the nodes
            if entity_uri not in node_ids.values:
                logger.warning(f"Entity URI {entity_uri} not found in the property graph")
                return {"related_entities": []}
            
            # Get all edges from the property graph
            all_edges = property_graph.edges
            edges_df = all_edges.to_pandas()
            
            # Convert to NetworkX-like graph structure with cugraph
            G = CuGraph()
            G.from_cudf_edgelist(
                all_edges,
                source="subject",
                destination="object",
                edge_attr="predicate"
            )
            
            # Run BFS from the entity_uri node
            bfs_result = cugraph.bfs(G, entity_uri, max_depth=max_distance)
            
            # Convert to pandas for processing
            bfs_df = bfs_result.to_pandas()
            
            # Filter results within the max_distance
            valid_distances = bfs_df[bfs_df["distance"] <= max_distance]
            valid_vertices = valid_distances[valid_distances["vertex"] != entity_uri]
            
            # Get the paths to each related entity
            related_entities = []
            
            for _, row in valid_vertices.iterrows():
                vertex = row["vertex"]
                distance = row["distance"]
                
                # Find edges between the source entity and this vertex
                # This is a simplified approach - in a real implementation we would
                # trace the full path from source to target
                
                # Find direct relationships
                direct_relations = edges_df[
                    ((edges_df["subject"] == entity_uri) & (edges_df["object"] == vertex)) |
                    ((edges_df["object"] == entity_uri) & (edges_df["subject"] == vertex))
                ]
                
                relation_types = []
                for _, rel in direct_relations.iterrows():
                    # Determine relation direction
                    if rel["subject"] == entity_uri:
                        direction = "outgoing"
                        relation = rel["predicate"]
                    else:
                        direction = "incoming"
                        relation = rel["predicate"]
                        
                    relation_types.append({
                        "predicate": relation,
                        "direction": direction
                    })
                
                # Add related entity to results
                related_entities.append({
                    "uri": vertex,
                    "distance": distance,
                    "direct_relations": relation_types
                })
            
            # Group results by relation type
            relation_groups = {}
            for entity in related_entities:
                for relation in entity.get("direct_relations", []):
                    predicate = relation["predicate"]
                    if predicate not in relation_groups:
                        relation_groups[predicate] = []
                    relation_groups[predicate].append(entity)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update stats
            self._stats["operations"]["find_related_entities"]["count"] += 1
            self._stats["operations"]["find_related_entities"]["total_time"] += elapsed_time
            
            logger.info(
                "Found %d related entities in %.2f seconds",
                len(related_entities),
                elapsed_time
            )
            
            return {
                "related_entities": related_entities,
                "relation_groups": relation_groups,
                "entity_uri": entity_uri,
                "max_distance": max_distance,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(
                "Error finding related entities: %s", str(e), exc_info=True
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update stats even for failed operations
            self._stats["operations"]["find_related_entities"]["count"] += 1
            self._stats["operations"]["find_related_entities"]["total_time"] += elapsed_time
            
            return {"related_entities": [], "error": str(e)}
    
    def run_sparql_query(self, property_graph: PropertyGraph, query: str) -> Dict[str, Any]:
        """
        Run a SPARQL-like query on the property graph using GPU acceleration.
        
        Args:
            property_graph: RAPIDS PropertyGraph
            query: SPARQL-like query string
            
        Returns:
            Query results
        """
        if not self.use_gpu or property_graph is None:
            logger.warning("RAPIDS acceleration not available for SPARQL queries")
            return {"results": []}
            
        try:
            # This is a placeholder for SPARQL query execution
            # In a real implementation, this would parse the SPARQL query and
            # translate it to RAPIDS graph operations
            
            logger.warning("SPARQL query execution not fully implemented yet")
            
            # Extract query type and patterns (very simplified)
            if "SELECT" in query:
                query_type = "SELECT"
            elif "CONSTRUCT" in query:
                query_type = "CONSTRUCT"
            else:
                query_type = "UNKNOWN"
                
            # Return placeholder results
            return {
                "query_type": query_type,
                "query": query,
                "results": [],
                "message": "SPARQL query execution not fully implemented"
            }
            
        except Exception as e:
            logger.error("Error executing SPARQL query: %s", str(e), exc_info=True)
            return {"results": [], "error": str(e)}
    
    def compute_graph_metrics(self, property_graph: PropertyGraph) -> Dict[str, Any]:
        """
        Compute metrics for the knowledge graph using GPU acceleration.
        
        Args:
            property_graph: RAPIDS PropertyGraph
            
        Returns:
            Dictionary of graph metrics
        """
        if not self.use_gpu or property_graph is None:
            logger.warning("RAPIDS acceleration not available for graph metrics")
            return {}
            
        try:
            # Get the cugraph from the property graph
            all_edges = property_graph.edges
            
            # Create a CuGraph for analytics
            G = CuGraph()
            G.from_cudf_edgelist(
                all_edges,
                source="subject",
                destination="object"
            )
            
            # Calculate basic metrics
            # Number of nodes and edges
            num_nodes = G.number_of_vertices()
            num_edges = G.number_of_edges()
            
            # Calculate connected components
            components = cugraph.connected_components(G)
            num_components = len(components["labels"].unique())
            
            # Compute PageRank to identify important entities
            pagerank = cugraph.pagerank(G)
            
            # Convert to pandas for processing
            pagerank_df = pagerank.to_pandas()
            
            # Get top 10 entities by PageRank
            top_entities = pagerank_df.nlargest(10, "pagerank")
            
            # Calculate graph diameter (approximate)
            # This is computationally expensive, so we use an estimation
            try:
                # Use a sample node for estimating diameter
                sample_vertex = G.nodes().iloc[0].item()
                
                # Run BFS from the sample node to estimate diameter
                bfs_result = cugraph.bfs(G, sample_vertex)
                estimated_diameter = bfs_result["distance"].max()
            except Exception as e:
                logger.warning(f"Error estimating graph diameter: {e}")
                estimated_diameter = -1
            
            # Return metrics
            return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_components": num_components,
                "estimated_diameter": estimated_diameter,
                "top_entities": top_entities.to_dict(orient="records"),
                "density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            }
            
        except Exception as e:
            logger.error("Error computing graph metrics: %s", str(e), exc_info=True)
            return {}
    
    def is_available(self) -> bool:
        """Check if RAPIDS acceleration is available."""
        return self.use_gpu and RAPIDS_AVAILABLE
    
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
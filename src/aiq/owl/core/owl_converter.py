"""
OWL converter module for transforming extracted financial data to OWL Turtle format.
"""
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import logging
from datetime import datetime
import uuid
import os
import json
from pathlib import Path

import rdflib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL, SKOS, DCTERMS, FOAF

# Try to import Owlready2 for enhanced OWL functionality
try:
    import owlready2
    from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty, World
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False

from aiq.owl.core.rapids_accelerator import RAPIDSAccelerator

logger = logging.getLogger(__name__)

# Define financial ontology namespaces
FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
FRO = Namespace("http://www.semanticweb.org/ontologies/fr-ontology#")
FINSIGHT = Namespace("http://finsight.dev/ontology/financial#")

# More detailed FIBO namespaces
FIBO_BE = Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/") # Business Entities
FIBO_FBC = Namespace("https://spec.edmcouncil.org/fibo/ontology/FBC/") # Financial Business and Commerce
FIBO_FND = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/") # FIBO Foundations
FIBO_IND = Namespace("https://spec.edmcouncil.org/fibo/ontology/IND/") # FIBO Indicators
FIBO_SEC = Namespace("https://spec.edmcouncil.org/fibo/ontology/SEC/") # Securities

class OwlConverter:
    """
    Converts extracted financial document data to OWL Turtle format.
    Uses RAPIDS acceleration when available and Owlready2 for enhanced OWL functionality.
    """
    
    def __init__(
        self,
        base_uri: str = "http://finsight.dev/kg/",
        include_provenance: bool = True,
        use_rapids: bool = True,
        rapids_memory_limit: Optional[int] = None,
        rapids_device_id: int = 0,
        ontology_path: Optional[str] = None,
        use_reasoner: bool = False,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the OWL converter.
        
        Args:
            base_uri: Base URI for the generated resources
            include_provenance: Whether to include provenance information
            use_rapids: Whether to use RAPIDS acceleration
            rapids_memory_limit: GPU memory limit for RAPIDS in bytes
            rapids_device_id: GPU device ID to use for RAPIDS
            ontology_path: Path to a custom ontology file (None for default)
            use_reasoner: Whether to use the Owlready2 reasoner for inference
            enable_cache: Whether to cache generated OWL files
            cache_dir: Directory for caching (None for default)
        """
        self.base_uri = base_uri
        self.include_provenance = include_provenance
        self.use_reasoner = use_reasoner and OWLREADY2_AVAILABLE
        
        # Initialize RDF graph with namespaces
        self.g = Graph()
        self._init_namespaces()
        
        # Initialize RAPIDS accelerator
        self.rapids = RAPIDSAccelerator(
            use_gpu=use_rapids,
            memory_limit=rapids_memory_limit,
            device_id=rapids_device_id,
            pool_allocator=True,
        )
        self.use_rapids = use_rapids and self.rapids.is_available()
        
        # Initialize Owlready2 world and ontology if available
        self.owlready2_available = OWLREADY2_AVAILABLE
        self.ontology = None
        self.world = None
        
        if OWLREADY2_AVAILABLE:
            try:
                self.world = World()
                
                # Load ontology if provided, otherwise create a new one
                if ontology_path and os.path.exists(ontology_path):
                    logger.info(f"Loading ontology from {ontology_path}")
                    self.ontology = self.world.get_ontology(f"file://{ontology_path}").load()
                    logger.info(f"Loaded ontology with {len(self.ontology.classes())} classes")
                else:
                    # Create a new ontology
                    logger.info("Creating new ontology")
                    self.ontology = self.world.get_ontology(f"{base_uri}ontology#")
                    
                    # Import FIBO ontology if available
                    try:
                        with self.ontology:
                            # Define basic classes for financial documents
                            class FinancialDocument(Thing): pass
                            class FinancialEntity(Thing): pass
                            class Section(Thing): pass
                            class Page(Thing): pass
                            class Table(Thing): pass
                            class TableHeader(Thing): pass
                            class TableRow(Thing): pass
                            class TableCell(Thing): pass
                            
                            # Define basic properties
                            class mentions(ObjectProperty): pass
                            class hasPage(ObjectProperty): domain = [FinancialDocument]; range = [Page]
                            class hasSection(ObjectProperty): domain = [Page]; range = [Section]
                            class hasTable(ObjectProperty): domain = [FinancialDocument]; range = [Table]
                            class hasRow(ObjectProperty): domain = [Table]; range = [TableRow]
                            class hasCell(ObjectProperty): domain = [TableRow]; range = [TableCell]
                            class hasHeader(ObjectProperty): domain = [Table]; range = [TableHeader]
                            class locatedOnPage(ObjectProperty): domain = [Table, Section]; range = [Page]
                            
                            # Data properties
                            class pageNumber(DataProperty): domain = [Page]; range = [int]
                            class content(DataProperty): range = [str]
                            class confidence(DataProperty): range = [float]
                            class entityType(DataProperty): domain = [FinancialEntity]; range = [str]
                            class x1(DataProperty): range = [float]
                            class y1(DataProperty): range = [float]
                            class x2(DataProperty): range = [float]
                            class y2(DataProperty): range = [float]
                            class rowIndex(DataProperty): domain = [TableRow]; range = [int]
                            class columnIndex(DataProperty): domain = [TableHeader, TableCell]; range = [int]
                            class cellValue(DataProperty): domain = [TableCell]; range = [str]
                    
                    except Exception as e:
                        logger.warning(f"Error setting up ontology classes: {e}")
                    
                # Check if reasoner can be used
                if self.use_reasoner:
                    try:
                        from owlready2 import sync_reasoner_pellet
                        logger.info("Pellet reasoner is available for inference")
                    except ImportError:
                        logger.warning("Pellet reasoner not available, disabling reasoning")
                        self.use_reasoner = False
                        
            except Exception as e:
                logger.error(f"Error initializing Owlready2: {e}")
                self.owlready2_available = False
        
        # Set up caching
        self.enable_cache = enable_cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".finsight", "cache", "owl")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all triples for batch processing
        self.triples = []
        
        # Mapping dictionaries for entity types and relationships
        self._init_entity_mappings()
        
        logger.info(
            "OwlConverter initialized with base_uri=%s, include_provenance=%s, use_rapids=%s, owlready2=%s",
            base_uri,
            include_provenance,
            self.use_rapids,
            self.owlready2_available,
        )
    
    def _init_namespaces(self) -> None:
        """Initialize namespaces for the RDF graph."""
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("xsd", XSD)
        self.g.bind("owl", OWL)
        self.g.bind("skos", SKOS)
        self.g.bind("dcterms", DCTERMS)
        self.g.bind("foaf", FOAF)
        self.g.bind("fibo", FIBO)
        self.g.bind("fibo-be", FIBO_BE)
        self.g.bind("fibo-fbc", FIBO_FBC)
        self.g.bind("fibo-fnd", FIBO_FND)
        self.g.bind("fibo-ind", FIBO_IND)
        self.g.bind("fibo-sec", FIBO_SEC)
        self.g.bind("fro", FRO)
        self.g.bind("finsight", FINSIGHT)
    
    def _init_entity_mappings(self) -> None:
        """Initialize entity type mappings for financial ontology."""
        # Detailed mapping from extracted entity types to ontology classes
        self.entity_type_mapping = {
            # Financial concepts
            "FINANCIAL_METRIC": FIBO_FBC.FinancialBusinessAndCommerce.FinancialMetric,
            "PROFIT": FIBO_FBC.FinancialBusinessAndCommerce.Profit,
            "REVENUE": FIBO_FBC.FinancialBusinessAndCommerce.Revenue,
            "ASSET": FIBO_FBC.FinancialBusinessAndCommerce.Asset,
            "LIABILITY": FIBO_FBC.FinancialBusinessAndCommerce.Liability,
            "DEBT": FIBO_FBC.FinancialBusinessAndCommerce.Debt,
            "EQUITY": FIBO_FBC.FinancialBusinessAndCommerce.Equity,
            "DIVIDEND": FIBO_FBC.FinancialBusinessAndCommerce.Dividend,
            "EXPENSE": FIBO_FBC.FinancialBusinessAndCommerce.Expense,
            "TAX": FIBO_FBC.FinancialBusinessAndCommerce.Tax,
            
            # Time and date concepts
            "TIME_PERIOD": FIBO_FND.DateAndTime.TimePeriod,
            "DATE": FIBO_FND.DateAndTime.Date,
            "YEAR": FIBO_FND.DateAndTime.Year,
            "QUARTER": FIBO_FND.DateAndTime.Quarter,
            "MONTH": FIBO_FND.DateAndTime.Month,
            
            # Money concepts
            "MONETARY_AMOUNT": FIBO_FND.Accounting.MonetaryAmount,
            "CURRENCY": FIBO_FND.Accounting.Currency,
            "EXCHANGE_RATE": FIBO_FND.Accounting.ExchangeRate,
            
            # Organization concepts
            "ORGANIZATION": FIBO_BE.LegalEntities.LegalEntity,
            "COMPANY": FIBO_BE.LegalEntities.Corporation,
            "BANK": FIBO_BE.LegalEntities.FinancialInstitution,
            
            # Person concepts
            "PERSON": FOAF.Person,
            "ROLE": FIBO_BE.Roles.Role,
            
            # Geographical concepts
            "LOCATION": FIBO_FND.Places.Location,
            "COUNTRY": FIBO_FND.Places.Country,
            "CITY": FIBO_FND.Places.City,
            "REGION": FIBO_FND.Places.Region,
            
            # Default
            "UNKNOWN": FINSIGHT.Entity,
        }
        
        # Entity property mappings for different entity types
        self.entity_property_mapping = {
            "FINANCIAL_METRIC": {
                "value": FIBO_FND.Accounting.hasMonetaryValue,
                "unit": FIBO_FND.Accounting.hasUnit,
                "currency": FIBO_FND.Accounting.hasCurrency,
                "period": FIBO_FND.DateAndTime.hasTimePeriod,
                "growth": FIBO_IND.Indicators.hasGrowthRate,
            },
            "ORGANIZATION": {
                "name": FOAF.name,
                "identifier": FIBO_BE.LegalEntities.hasLegalEntityIdentifier,
                "foundation_date": FIBO_BE.LegalEntities.hasFoundingDate,
                "jurisdiction": FIBO_BE.LegalEntities.hasJurisdiction,
            },
            "PERSON": {
                "name": FOAF.name,
                "title": FOAF.title,
                "role": FIBO_BE.Roles.hasRole,
            },
        }
    
    def _add_triple(self, subject: URIRef, predicate: URIRef, obj: Any) -> None:
        """
        Add a triple to the graph and the triples list.
        
        Args:
            subject: Subject of the triple
            predicate: Predicate of the triple
            obj: Object of the triple
        """
        # Add to graph directly
        self.g.add((subject, predicate, obj))
        
        # Also add to triples list for batch processing
        if self.use_rapids:
            # Convert to strings for RAPIDS processing
            self.triples.append((str(subject), str(predicate), str(obj)))
        
        # Add to Owlready2 ontology if available
        if self.owlready2_available and self.ontology:
            try:
                # This would be implemented for a production system
                # It requires translating RDFlib triples to Owlready2 objects
                pass
            except Exception as e:
                logger.debug(f"Error adding triple to Owlready2 ontology: {e}")
    
    def convert(self, document_data: Dict[str, Any]) -> Graph:
        """
        Convert extracted document data to OWL.
        
        Args:
            document_data: Structured data extracted from the document
            
        Returns:
            RDF graph in OWL format
        """
        # Clear any existing data
        self.clear()
        
        document_id = document_data.get("document_id", f"doc_{uuid.uuid4().hex[:8]}")
        document_uri = URIRef(f"{self.base_uri}document/{document_id}")
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(document_data)
            cache_file = self.cache_dir / f"{cache_key}.ttl"
            
            if cache_file.exists():
                try:
                    # Load from cache
                    logger.info(f"Loading from cache: {cache_file}")
                    self.g.parse(str(cache_file), format="turtle")
                    
                    # Add minimal provenance information about cache
                    if self.include_provenance:
                        prov_uri = URIRef(f"{document_uri}/provenance")
                        self._add_triple(prov_uri, DCTERMS.source, Literal("cache"))
                        self._add_triple(prov_uri, DCTERMS.modified, 
                                       Literal(datetime.now().isoformat(), datatype=XSD.dateTime))
                    
                    return self.g
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")
        
        # Add document metadata
        self._add_document_metadata(document_uri, document_data.get("metadata", {}))
        
        # Add document structure
        self._add_document_structure(document_uri, document_data.get("layout", []))
        
        # Add tables
        self._add_tables(document_uri, document_data.get("tables", []))
        
        # Add financial entities
        self._add_financial_entities(document_uri, document_data.get("entities", []))
        
        # Establish relationships between entities
        self._link_entities(document_uri, document_data)
        
        # Add provenance information if requested
        if self.include_provenance:
            self._add_provenance(document_uri)
        
        # Apply reasoning if requested and available
        if self.use_reasoner and self.owlready2_available and self.ontology:
            try:
                self._apply_reasoning()
            except Exception as e:
                logger.error(f"Error applying reasoning: {e}", exc_info=True)
        
        # Process with RAPIDS if available
        if self.use_rapids and self.triples:
            try:
                # Use RAPIDS to accelerate graph creation
                logger.info("Using RAPIDS to accelerate graph generation")
                rapids_graph = self.rapids.accelerate_owl_generation(self.triples, self.base_uri)
                
                # If successful, replace our graph with the RAPIDS-generated one
                if rapids_graph is not None and len(rapids_graph) > 0:
                    # Preserve namespace bindings
                    for prefix, namespace in self.g.namespaces():
                        rapids_graph.bind(prefix, namespace)
                    
                    self.g = rapids_graph
                    logger.info("Successfully used RAPIDS for graph generation")
            except Exception as e:
                logger.error("Error using RAPIDS for graph generation: %s", str(e), exc_info=True)
                # Continue with the graph we've built normally
        
        # Cache the result if enabled
        if self.enable_cache:
            try:
                cache_key = self._generate_cache_key(document_data)
                cache_file = self.cache_dir / f"{cache_key}.ttl"
                
                # Save to cache
                logger.info(f"Saving to cache: {cache_file}")
                self.g.serialize(destination=str(cache_file), format="turtle")
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")
        
        logger.info(
            "Converted document %s to OWL with %s triples",
            document_id,
            len(self.g),
        )
        
        return self.g
    
    def _generate_cache_key(self, document_data: Dict[str, Any]) -> str:
        """
        Generate a cache key for the document data.
        
        Args:
            document_data: Document data to generate a key for
            
        Returns:
            Cache key as a string
        """
        # Use document ID if available
        if "document_id" in document_data:
            document_id = document_data["document_id"]
            return f"doc_{document_id}"
        
        # Otherwise, generate a hash of essential data
        import hashlib
        
        # Create a stable representation of the document data for hashing
        hash_data = {}
        
        # Include metadata
        if "metadata" in document_data and isinstance(document_data["metadata"], dict):
            hash_data["metadata"] = {
                k: v for k, v in document_data["metadata"].items() 
                if k in ["title", "author", "creation_date", "page_count"]
            }
        
        # Include entity texts
        if "entities" in document_data and isinstance(document_data["entities"], list):
            hash_data["entities"] = [
                entity.get("text", "") for entity in document_data["entities"]
            ]
        
        # Calculate hash
        hash_str = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()
        return f"doc_{hash_str[:16]}"
    
    def _apply_reasoning(self) -> None:
        """Apply OWL reasoning to infer additional triples."""
        if not self.use_reasoner or not self.owlready2_available or not self.ontology:
            return
            
        try:
            logger.info("Applying OWL reasoning")
            start_time = datetime.now()
            
            # Sync the RDFlib graph to Owlready2
            for s, p, o in self.g:
                # TODO: Implement proper syncing between RDFlib and Owlready2
                pass
            
            # Apply the reasoner
            from owlready2 import sync_reasoner_pellet
            sync_reasoner_pellet(self.world)
            
            # Extract inferred triples
            inferred_count = 0  # Track number of inferred triples
            # TODO: Implement extraction of inferred triples from Owlready2 to RDFlib
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"Reasoning completed in {elapsed:.2f} seconds, added {inferred_count} inferred triples")
            
        except Exception as e:
            logger.error(f"Error during reasoning: {e}", exc_info=True)
    
    def _add_document_metadata(self, document_uri: URIRef, metadata: Dict[str, Any]) -> None:
        """
        Add document metadata to the graph with appropriate ontology mappings.
        
        Args:
            document_uri: URI reference for the document
            metadata: Dictionary of document metadata
        """
        # Document type - use more specific type if we can determine it
        document_type = FINSIGHT.FinancialDocument
        
        # Try to determine more specific document type
        title = metadata.get("title", "").lower()
        if title:
            if "annual report" in title or "10-k" in title:
                document_type = FIBO_SEC.Reporting.AnnualReport
            elif "quarterly report" in title or "10-q" in title:
                document_type = FIBO_SEC.Reporting.QuarterlyReport
            elif "financial statement" in title:
                document_type = FIBO_FBC.FinancialReport.FinancialStatement
            elif "balance sheet" in title:
                document_type = FIBO_FBC.FinancialReport.BalanceSheet
            elif "income statement" in title or "profit and loss" in title:
                document_type = FIBO_FBC.FinancialReport.IncomeStatement
            elif "cash flow" in title:
                document_type = FIBO_FBC.FinancialReport.CashFlowStatement
            elif "prospectus" in title:
                document_type = FIBO_SEC.SecuritiesIssuance.Prospectus
        
        # Add document type
        self._add_triple(document_uri, RDF.type, document_type)
        
        # Basic Dublin Core metadata
        if metadata.get("title"):
            self._add_triple(document_uri, DCTERMS.title, Literal(metadata["title"]))
        if metadata.get("subject"):
            self._add_triple(document_uri, DCTERMS.subject, Literal(metadata["subject"]))
        if metadata.get("author"):
            self._add_triple(document_uri, DCTERMS.creator, Literal(metadata["author"]))
            
            # Also add as FOAF creator if it looks like an organization or person
            creator_uri = URIRef(f"{document_uri}/creator")
            self._add_triple(document_uri, FOAF.maker, creator_uri)
            if "inc" in metadata["author"].lower() or "corp" in metadata["author"].lower() or "ltd" in metadata["author"].lower():
                self._add_triple(creator_uri, RDF.type, FIBO_BE.LegalEntities.Corporation)
            else:
                self._add_triple(creator_uri, RDF.type, FOAF.Person)
            self._add_triple(creator_uri, FOAF.name, Literal(metadata["author"]))
            
        # Date information with proper XSD types
        if metadata.get("creation_date"):
            try:
                # Try to parse date to ensure correct format
                creation_date = self._parse_date(metadata["creation_date"])
                self._add_triple(document_uri, DCTERMS.created, Literal(creation_date, datatype=XSD.dateTime))
            except:
                # If parsing fails, use as string
                self._add_triple(document_uri, DCTERMS.created, Literal(metadata["creation_date"]))
        
        if metadata.get("modification_date"):
            try:
                # Try to parse date to ensure correct format
                mod_date = self._parse_date(metadata["modification_date"])
                self._add_triple(document_uri, DCTERMS.modified, Literal(mod_date, datatype=XSD.dateTime))
            except:
                # If parsing fails, use as string
                self._add_triple(document_uri, DCTERMS.modified, Literal(metadata["modification_date"]))
        
        # Document properties with appropriate datatypes
        self._add_triple(document_uri, FINSIGHT.pageCount, 
                       Literal(metadata.get("page_count", 0), datatype=XSD.integer))
        
        # Add keywords if available
        if metadata.get("keywords"):
            keywords = metadata["keywords"]
            if isinstance(keywords, str):
                # Split comma-separated keywords
                for keyword in [k.strip() for k in keywords.split(",") if k.strip()]:
                    self._add_triple(document_uri, DCTERMS.subject, Literal(keyword))
            elif isinstance(keywords, list):
                for keyword in keywords:
                    self._add_triple(document_uri, DCTERMS.subject, Literal(keyword))
        
        # Add publisher information if available
        if metadata.get("producer") or metadata.get("creator"):
            publisher = metadata.get("producer") or metadata.get("creator")
            self._add_triple(document_uri, DCTERMS.publisher, Literal(publisher))
    
    def _parse_date(self, date_str: str) -> str:
        """
        Parse a date string into a standardized ISO format.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            ISO formatted date string
        """
        # Handle common PDF date format: "D:YYYYMMDDHHmmSS"
        if date_str.startswith("D:"):
            date_str = date_str[2:]
            
            # Basic format: YYYYMMDDHHmmSS
            if len(date_str) >= 14:
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                hour = date_str[8:10]
                minute = date_str[10:12]
                second = date_str[12:14]
                return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
            # Partial date
            elif len(date_str) >= 8:
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{year}-{month}-{day}T00:00:00"
        
        # If we can't parse it, return as is
        return date_str
    
    def _add_document_structure(self, document_uri: URIRef, layout: List[Dict[str, Any]]) -> None:
        """Add document structure to the graph."""
        for page_layout in layout:
            page_number = page_layout.get("page_number", 0)
            page_uri = URIRef(f"{document_uri}/page/{page_number}")
            
            # Page type
            self._add_triple(page_uri, RDF.type, FINSIGHT.Page)
            self._add_triple(page_uri, FINSIGHT.pageNumber, 
                           Literal(page_number, datatype=XSD.integer))
            self._add_triple(document_uri, FINSIGHT.hasPage, page_uri)
            
            # Add sections
            for idx, section in enumerate(page_layout.get("sections", [])):
                section_uri = URIRef(f"{page_uri}/section/{idx}")
                
                # Section type
                self._add_triple(section_uri, RDF.type, FINSIGHT.Section)
                self._add_triple(section_uri, FINSIGHT.sectionType, Literal(section["type"]))
                self._add_triple(section_uri, FINSIGHT.content, Literal(section["text"]))
                self._add_triple(page_uri, FINSIGHT.hasSection, section_uri)
                
                # Add bounding box if available
                if "bbox" in section:
                    bbox = section["bbox"]
                    self._add_triple(section_uri, FINSIGHT.x1, Literal(bbox[0], datatype=XSD.float))
                    self._add_triple(section_uri, FINSIGHT.y1, Literal(bbox[1], datatype=XSD.float))
                    self._add_triple(section_uri, FINSIGHT.x2, Literal(bbox[2], datatype=XSD.float))
                    self._add_triple(section_uri, FINSIGHT.y2, Literal(bbox[3], datatype=XSD.float))
                
                # Add confidence if available
                if "confidence" in section:
                    self._add_triple(section_uri, FINSIGHT.confidence, 
                                   Literal(section["confidence"], datatype=XSD.float))
    
    def _add_tables(self, document_uri: URIRef, tables: List[Dict[str, Any]]) -> None:
        """Add tables to the graph."""
        for table in tables:
            table_id = table.get("table_id", f"table_{uuid.uuid4().hex[:8]}")
            table_uri = URIRef(f"{document_uri}/table/{table_id}")
            
            # Table type
            self._add_triple(table_uri, RDF.type, FINSIGHT.Table)
            self._add_triple(document_uri, FINSIGHT.hasTable, table_uri)
            
            # Table metadata
            if "page_number" in table:
                page_uri = URIRef(f"{document_uri}/page/{table['page_number']}")
                self._add_triple(table_uri, FINSIGHT.locatedOnPage, page_uri)
            
            # Add bounding box if available
            if "bbox" in table:
                bbox = table["bbox"]
                self._add_triple(table_uri, FINSIGHT.x1, Literal(bbox[0], datatype=XSD.float))
                self._add_triple(table_uri, FINSIGHT.y1, Literal(bbox[1], datatype=XSD.float))
                self._add_triple(table_uri, FINSIGHT.x2, Literal(bbox[2], datatype=XSD.float))
                self._add_triple(table_uri, FINSIGHT.y2, Literal(bbox[3], datatype=XSD.float))
            
            # Table headers
            headers = table.get("headers", [])
            for col_idx, header in enumerate(headers):
                header_uri = URIRef(f"{table_uri}/header/{col_idx}")
                self._add_triple(header_uri, RDF.type, FINSIGHT.TableHeader)
                self._add_triple(header_uri, FINSIGHT.columnIndex, 
                               Literal(col_idx, datatype=XSD.integer))
                self._add_triple(header_uri, RDFS.label, Literal(header))
                self._add_triple(table_uri, FINSIGHT.hasHeader, header_uri)
            
            # Table data rows
            data_rows = table.get("data", [])
            for row_idx, row in enumerate(data_rows):
                row_uri = URIRef(f"{table_uri}/row/{row_idx}")
                self._add_triple(row_uri, RDF.type, FINSIGHT.TableRow)
                self._add_triple(row_uri, FINSIGHT.rowIndex, Literal(row_idx, datatype=XSD.integer))
                self._add_triple(table_uri, FINSIGHT.hasRow, row_uri)
                
                # Cells in the row
                for col_idx, cell_value in enumerate(row):
                    cell_uri = URIRef(f"{row_uri}/cell/{col_idx}")
                    self._add_triple(cell_uri, RDF.type, FINSIGHT.TableCell)
                    self._add_triple(cell_uri, FINSIGHT.columnIndex, 
                                   Literal(col_idx, datatype=XSD.integer))
                    self._add_triple(cell_uri, FINSIGHT.cellValue, Literal(cell_value))
                    self._add_triple(row_uri, FINSIGHT.hasCell, cell_uri)
                    
                    # Link cell to header
                    if col_idx < len(headers):
                        header_uri = URIRef(f"{table_uri}/header/{col_idx}")
                        self._add_triple(cell_uri, FINSIGHT.hasHeader, header_uri)
    
    def _add_financial_entities(self, document_uri: URIRef, entities: List[Dict[str, Any]]) -> None:
        """
        Add financial entities to the graph with proper ontology mappings.
        
        Args:
            document_uri: Document URI reference
            entities: List of extracted entities
        """
        entity_map = {}  # Store entity URIs for later relation linking
        
        for idx, entity in enumerate(entities):
            # Generate a stable ID based on entity text and type
            entity_text = entity.get("text", "").strip()
            entity_type = entity.get("type", "UNKNOWN")
            
            if not entity_text:
                continue
                
            # Generate more specific ID for well-known entities
            if entity_type == "ORGANIZATION" and len(entity_text) > 2:
                # Use normalized organization name as part of the ID
                clean_name = entity_text.lower().replace(" ", "_").replace(".", "").replace(",", "")
                entity_id = f"org_{clean_name}_{idx}"
            elif entity_type == "PERSON" and len(entity_text) > 2:
                # Use normalized person name as part of the ID
                clean_name = entity_text.lower().replace(" ", "_").replace(".", "").replace(",", "")
                entity_id = f"person_{clean_name}_{idx}"
            elif entity_type == "FINANCIAL_METRIC" and len(entity_text) > 2:
                # Use normalized metric name
                clean_name = entity_text.lower().replace(" ", "_").replace(".", "").replace(",", "")
                entity_id = f"metric_{clean_name}_{idx}"
            else:
                # Default ID format
                entity_id = f"entity_{entity_type.lower()}_{idx}"
                
            entity_uri = URIRef(f"{document_uri}/entity/{entity_id}")
            
            # Store URI for linking
            entity["_uri"] = entity_uri
            entity_map[idx] = entity_uri
            
            # Get appropriate ontology type from mapping or default
            ontology_type = self.entity_type_mapping.get(entity_type, FINSIGHT.Entity)
            
            # Add entity type
            self._add_triple(entity_uri, RDF.type, ontology_type)
            
            # For unknown types, add the type string as well
            if ontology_type == FINSIGHT.Entity:
                self._add_triple(entity_uri, FINSIGHT.entityType, Literal(entity_type))
            
            # Add basic properties
            self._add_triple(entity_uri, RDFS.label, Literal(entity_text))
            self._add_triple(document_uri, FINSIGHT.mentions, entity_uri)
            
            # Add SKOS preferred label for better search
            self._add_triple(entity_uri, SKOS.prefLabel, Literal(entity_text))
            
            # Entity confidence if available
            if "confidence" in entity:
                conf_value = float(entity["confidence"])  # Ensure it's a float
                self._add_triple(entity_uri, FINSIGHT.confidence, 
                               Literal(conf_value, datatype=XSD.float))
            
            # Add text offsets if available
            if "start_offset" in entity and "end_offset" in entity:
                self._add_triple(entity_uri, FINSIGHT.startOffset, 
                               Literal(entity["start_offset"], datatype=XSD.integer))
                self._add_triple(entity_uri, FINSIGHT.endOffset, 
                               Literal(entity["end_offset"], datatype=XSD.integer))
            
            # Add context if available
            if "context" in entity:
                self._add_triple(entity_uri, FINSIGHT.context, Literal(entity["context"]))
            
            # Add entity-specific properties
            self._add_entity_properties(entity_uri, entity, entity_type)
    
    def _add_entity_properties(self, entity_uri: URIRef, entity: Dict[str, Any], entity_type: str) -> None:
        """
        Add specific properties for different entity types.
        
        Args:
            entity_uri: URI reference for the entity
            entity: Entity data dictionary
            entity_type: Type of the entity
        """
        # Get property mappings for this entity type
        property_mapping = self.entity_property_mapping.get(entity_type, {})
        
        # Extract properties from context using NLP patterns (simplified)
        context = entity.get("context", "")
        text = entity.get("text", "")
        
        # Special handling for financial metrics
        if entity_type == "FINANCIAL_METRIC" or entity_type == "MONETARY_AMOUNT":
            # Try to extract numeric value if present in the text
            import re
            
            # Look for numeric patterns in the text or context
            numeric_pattern = r'([\$€£¥]?\s*\d+(?:[.,]\d+)?(?:\s*[bBmMkK]illion)?)'
            currency_pattern = r'(USD|EUR|GBP|JPY|CNY|\$|€|£|¥)'
            period_pattern = r'(20\d{2}|Q[1-4]|[fF]iscal\s+[yY]ear|[fF]Y\s*\d{2,4}|[Qq]uarter(?:\s+\d)?)'
            
            # Extract value
            value_match = re.search(numeric_pattern, text) or re.search(numeric_pattern, context)
            if value_match:
                value_text = value_match.group(1).strip()
                
                # Clean and normalize value
                value_text = value_text.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").strip()
                
                # Handle scale terms
                scale = 1.0
                if "billion" in value_text.lower() or "b" in value_text.lower():
                    scale = 1_000_000_000
                    value_text = value_text.lower().replace("billion", "").replace("b", "").strip()
                elif "million" in value_text.lower() or "m" in value_text.lower():
                    scale = 1_000_000
                    value_text = value_text.lower().replace("million", "").replace("m", "").strip()
                elif "thousand" in value_text.lower() or "k" in value_text.lower():
                    scale = 1_000
                    value_text = value_text.lower().replace("thousand", "").replace("k", "").strip()
                
                # Convert to numeric value
                try:
                    numeric_value = float(value_text.replace(",", "")) * scale
                    if numeric_value.is_integer():
                        self._add_triple(entity_uri, property_mapping.get("value", FIBO_FND.Accounting.hasMonetaryValue), 
                                      Literal(int(numeric_value), datatype=XSD.integer))
                    else:
                        self._add_triple(entity_uri, property_mapping.get("value", FIBO_FND.Accounting.hasMonetaryValue), 
                                      Literal(numeric_value, datatype=XSD.decimal))
                except ValueError:
                    # If parsing fails, just add as a string
                    self._add_triple(entity_uri, property_mapping.get("value", FIBO_FND.Accounting.hasMonetaryValue), 
                                  Literal(value_text))
            
            # Extract currency
            currency_match = re.search(currency_pattern, text) or re.search(currency_pattern, context)
            if currency_match:
                currency = currency_match.group(1)
                # Map currency symbols to codes
                currency_map = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
                currency_code = currency_map.get(currency, currency)
                
                # Create currency entity
                currency_uri = URIRef(f"{self.base_uri}currency/{currency_code}")
                self._add_triple(currency_uri, RDF.type, FIBO_FND.Accounting.Currency)
                self._add_triple(currency_uri, SKOS.prefLabel, Literal(currency_code))
                self._add_triple(entity_uri, property_mapping.get("currency", FIBO_FND.Accounting.hasCurrency), currency_uri)
            
            # Extract time period
            period_match = re.search(period_pattern, text) or re.search(period_pattern, context)
            if period_match:
                period_text = period_match.group(1)
                period_uri = URIRef(f"{entity_uri}/period")
                self._add_triple(period_uri, RDF.type, FIBO_FND.DateAndTime.TimePeriod)
                self._add_triple(period_uri, RDFS.label, Literal(period_text))
                self._add_triple(entity_uri, property_mapping.get("period", FIBO_FND.DateAndTime.hasTimePeriod), period_uri)
        
        # Special handling for organizations
        elif entity_type == "ORGANIZATION" or entity_type == "COMPANY":
            # Create or reference a canonical entity for the organization
            org_name = text.strip()
            canonical_uri = URIRef(f"{self.base_uri}organization/{org_name.lower().replace(' ', '_')}")
            
            # Link document entity to canonical entity
            self._add_triple(entity_uri, OWL.sameAs, canonical_uri)
            
            # Add organization properties
            self._add_triple(canonical_uri, RDF.type, FIBO_BE.LegalEntities.LegalEntity)
            self._add_triple(canonical_uri, FOAF.name, Literal(org_name))
            
            # Extract organization info from context if available
            if context:
                # Very basic extraction - would be more sophisticated in real implementation
                if "founded in" in context.lower() or "established in" in context.lower():
                    import re
                    year_match = re.search(r'(founded|established)\s+in\s+(\d{4})', context.lower())
                    if year_match:
                        year = year_match.group(2)
                        self._add_triple(canonical_uri, FIBO_BE.LegalEntities.hasFoundingDate,
                                       Literal(f"{year}-01-01", datatype=XSD.date))
                
                # Check for location mentions
                location_pattern = r'based\s+in\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)'
                location_match = re.search(location_pattern, context)
                if location_match:
                    location_name = location_match.group(1)
                    location_uri = URIRef(f"{self.base_uri}location/{location_name.lower().replace(' ', '_')}")
                    self._add_triple(location_uri, RDF.type, FIBO_FND.Places.Location)
                    self._add_triple(location_uri, RDFS.label, Literal(location_name))
                    self._add_triple(canonical_uri, FIBO_BE.LegalEntities.hasHeadquartersAddress, location_uri)
        
        # Special handling for persons
        elif entity_type == "PERSON":
            person_name = text.strip()
            
            # Extract title if present (Mr., Mrs., Dr., etc.)
            import re
            title_match = re.search(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+', person_name)
            if title_match:
                title = title_match.group(1)
                self._add_triple(entity_uri, FOAF.title, Literal(title))
                
                # Remove title from name for other properties
                person_name = person_name[len(title):].strip()
            
            # Add name properties
            self._add_triple(entity_uri, FOAF.name, Literal(person_name))
            
            # Try to extract given name and family name if possible
            name_parts = person_name.split()
            if len(name_parts) > 1:
                given_name = name_parts[0]
                family_name = name_parts[-1]
                self._add_triple(entity_uri, FOAF.givenName, Literal(given_name))
                self._add_triple(entity_uri, FOAF.familyName, Literal(family_name))
            
            # Extract role information from context if available
            if context:
                role_pattern = r'(?:[,\s]|^)(CEO|CFO|CTO|Chairman|Director|President|Manager|Executive|Officer)'
                role_match = re.search(role_pattern, context, re.IGNORECASE)
                if role_match:
                    role = role_match.group(1)
                    role_uri = URIRef(f"{self.base_uri}role/{role.lower()}")
                    self._add_triple(role_uri, RDF.type, FIBO_BE.Roles.Role)
                    self._add_triple(role_uri, RDFS.label, Literal(role))
                    self._add_triple(entity_uri, FIBO_BE.Roles.hasRole, role_uri)
    
    def _link_entities(self, document_uri: URIRef, document_data: Dict[str, Any]) -> None:
        """
        Establish relationships between entities in the document.
        
        Args:
            document_uri: URI reference for the document
            document_data: Complete document data with entities and structure
        """
        entities = document_data.get("entities", [])
        
        # Skip if no entities
        if not entities:
            return
            
        # Extract all entity URIs first
        entity_map = {}  # Map from index to URI
        for idx, entity in enumerate(entities):
            if "_uri" in entity:
                entity_map[idx] = entity["_uri"]
        
        # Find and link related entities based on context proximity
        for i, entity1 in enumerate(entities):
            if i not in entity_map:
                continue
                
            entity1_uri = entity_map[i]
            entity1_type = entity1.get("type", "UNKNOWN")
            
            # Get entity start and end positions
            start1 = entity1.get("start_offset", -1)
            end1 = entity1.get("end_offset", -1)
            
            # Skip entities without position info
            if start1 < 0 or end1 < 0:
                continue
                
            # Look for related entities
            for j, entity2 in enumerate(entities):
                if i == j or j not in entity_map:
                    continue
                    
                entity2_uri = entity_map[j]
                entity2_type = entity2.get("type", "UNKNOWN")
                
                # Get entity start and end positions
                start2 = entity2.get("start_offset", -1)
                end2 = entity2.get("end_offset", -1)
                
                # Skip entities without position info
                if start2 < 0 or end2 < 0:
                    continue
                
                # Check if entities are in close proximity (within 100 characters)
                proximity = min(abs(start1 - end2), abs(start2 - end1))
                if proximity <= 100:
                    # Create relationship based on entity types
                    self._create_entity_relationship(entity1_uri, entity1_type, entity2_uri, entity2_type)
    
    def _create_entity_relationship(
        self, 
        entity1_uri: URIRef, 
        entity1_type: str, 
        entity2_uri: URIRef, 
        entity2_type: str
    ) -> None:
        """
        Create appropriate relationship between two entities.
        
        Args:
            entity1_uri: URI of the first entity
            entity1_type: Type of the first entity
            entity2_uri: URI of the second entity
            entity2_type: Type of the second entity
        """
        # Define relationship patterns
        if entity1_type == "ORGANIZATION" and entity2_type == "FINANCIAL_METRIC":
            # Organization has financial metric
            self._add_triple(entity1_uri, FIBO_FBC.FinancialBusinessAndCommerce.hasFinancialMetric, entity2_uri)
            
        elif entity1_type == "FINANCIAL_METRIC" and entity2_type == "ORGANIZATION":
            # Financial metric belongs to organization
            self._add_triple(entity2_uri, FIBO_FBC.FinancialBusinessAndCommerce.hasFinancialMetric, entity1_uri)
            
        elif entity1_type == "FINANCIAL_METRIC" and entity2_type == "TIME_PERIOD":
            # Financial metric for time period
            self._add_triple(entity1_uri, FIBO_FND.DateAndTime.hasTimePeriod, entity2_uri)
            
        elif entity1_type == "TIME_PERIOD" and entity2_type == "FINANCIAL_METRIC":
            # Time period has financial metric
            self._add_triple(entity2_uri, FIBO_FND.DateAndTime.hasTimePeriod, entity1_uri)
            
        elif entity1_type == "PERSON" and entity2_type == "ORGANIZATION":
            # Person affiliated with organization
            self._add_triple(entity1_uri, FIBO_BE.Roles.isAffiliatedWith, entity2_uri)
            
        elif entity1_type == "ORGANIZATION" and entity2_type == "PERSON":
            # Organization has affiliated person
            self._add_triple(entity2_uri, FIBO_BE.Roles.isAffiliatedWith, entity1_uri)
            
        elif entity1_type == "ORGANIZATION" and entity2_type == "LOCATION":
            # Organization located in place
            self._add_triple(entity1_uri, FIBO_FND.Places.hasLocation, entity2_uri)
            
        elif entity1_type == "LOCATION" and entity2_type == "ORGANIZATION":
            # Location contains organization
            self._add_triple(entity2_uri, FIBO_FND.Places.hasLocation, entity1_uri)
            
        else:
            # Generic relationship for other type combinations
            self._add_triple(entity1_uri, FINSIGHT.relatedTo, entity2_uri)
    
    def _add_provenance(self, document_uri: URIRef) -> None:
        """
        Add detailed provenance information to the graph.
        
        Args:
            document_uri: URI reference for the document
        """
        # Create provenance node
        prov_uri = URIRef(f"{document_uri}/provenance")
        self._add_triple(prov_uri, RDF.type, DCTERMS.ProvenanceStatement)
        self._add_triple(document_uri, DCTERMS.provenance, prov_uri)
        
        # Add extraction details with proper timestamps
        timestamp = datetime.now().isoformat()
        self._add_triple(prov_uri, DCTERMS.created, Literal(timestamp, datatype=XSD.dateTime))
        
        # Add software agent information
        agent_uri = URIRef(f"{self.base_uri}agent/FinSightOWLConverter")
        self._add_triple(agent_uri, RDF.type, FOAF.Agent)
        self._add_triple(agent_uri, FOAF.name, Literal("FinSight OWL Converter"))
        self._add_triple(agent_uri, DCTERMS.description, 
                     Literal("NVIDIA-accelerated financial document to OWL converter"))
        self._add_triple(agent_uri, OWL.versionInfo, Literal("1.0.0"))
        
        # Link agent to provenance
        self._add_triple(prov_uri, DCTERMS.creator, agent_uri)
        
        # Add extraction system details
        system_uri = URIRef(f"{self.base_uri}system/NVIDIAFinancialPDFToOWLBlueprint")
        self._add_triple(system_uri, RDF.type, FINSIGHT.ExtractionSystem)
        self._add_triple(system_uri, RDFS.label, Literal("NVIDIA Financial PDF to OWL Blueprint"))
        self._add_triple(prov_uri, FINSIGHT.extractionSystem, system_uri)
        
        # Add RAPIDS usage information with details
        self._add_triple(prov_uri, FINSIGHT.usedRAPIDSAcceleration, Literal(self.use_rapids, datatype=XSD.boolean))
        
        if self.use_rapids:
            # Add RAPIDS details
            rapids_uri = URIRef(f"{self.base_uri}technology/RAPIDS")
            self._add_triple(rapids_uri, RDF.type, FINSIGHT.AccelerationTechnology)
            self._add_triple(rapids_uri, RDFS.label, Literal("NVIDIA RAPIDS"))
            self._add_triple(prov_uri, FINSIGHT.usedAccelerationTechnology, rapids_uri)
            
            # Add performance metrics if available
            stats = self.rapids.get_stats()
            if stats and isinstance(stats, dict):
                metrics_uri = URIRef(f"{prov_uri}/performance")
                self._add_triple(metrics_uri, RDF.type, FINSIGHT.PerformanceMetrics)
                self._add_triple(prov_uri, FINSIGHT.hasPerformanceMetrics, metrics_uri)
                
                # Add basic metrics
                if "gpu_info" in stats:
                    gpu_info = stats["gpu_info"]
                    if "name" in gpu_info:
                        self._add_triple(metrics_uri, FINSIGHT.gpuName, Literal(gpu_info["name"]))
                    if "total_memory" in gpu_info:
                        self._add_triple(metrics_uri, FINSIGHT.gpuTotalMemory, 
                                      Literal(gpu_info["total_memory"], datatype=XSD.integer))
                
                # Add operation metrics
                if "operations" in stats:
                    ops = stats["operations"]
                    for op_name, op_stats in ops.items():
                        if "avg_time" in op_stats:
                            self._add_triple(metrics_uri, FINSIGHT[f"{op_name}AvgTime"], 
                                         Literal(op_stats["avg_time"], datatype=XSD.float))
        
        # Add Owlready2 usage information
        self._add_triple(prov_uri, FINSIGHT.usedOwlready2, Literal(self.owlready2_available, datatype=XSD.boolean))
        
        if self.owlready2_available:
            # Add reasoning details
            self._add_triple(prov_uri, FINSIGHT.usedReasoning, Literal(self.use_reasoner, datatype=XSD.boolean))
    
    def convert_to_property_graph(self) -> Any:
        """
        Convert the RDF graph to a RAPIDS PropertyGraph for accelerated querying.
        
        Returns:
            RAPIDS PropertyGraph or None if RAPIDS is not available
        """
        if not self.use_rapids:
            logger.warning("RAPIDS acceleration not available for property graph conversion")
            return None
        
        return self.rapids.rdf_to_property_graph(self.g)
    
    def find_related_entities(self, entity_uri: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Find entities related to the given entity within a maximum distance.
        
        Args:
            entity_uri: URI of the entity to find related entities for
            max_distance: Maximum distance (number of hops) to search
            
        Returns:
            Dictionary of related entities grouped by relationship type
        """
        if not self.use_rapids:
            logger.warning("RAPIDS acceleration not available for entity relation finding")
            return self._fallback_find_related_entities(entity_uri, max_distance)
        
        # Convert to PropertyGraph
        property_graph = self.convert_to_property_graph()
        
        # Find related entities
        return self.rapids.find_related_entities(property_graph, entity_uri, max_distance)
    
    def _fallback_find_related_entities(self, entity_uri: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        CPU fallback implementation for finding related entities.
        
        Args:
            entity_uri: URI of the entity to find related entities for
            max_distance: Maximum distance (number of hops) to search
            
        Returns:
            Dictionary of related entities
        """
        # Use RDFlib graph querying capabilities as fallback
        logger.info(f"Using CPU fallback to find entities related to {entity_uri}")
        related_entities = []
        
        # Convert string URI to URIRef if needed
        if not isinstance(entity_uri, URIRef):
            entity_uri = URIRef(entity_uri)
        
        # Find directly related entities (distance 1)
        # Outgoing relations
        for s, p, o in self.g.triples((entity_uri, None, None)):
            if isinstance(o, URIRef) and o != entity_uri:
                # Get entity type and label
                entity_type = "Unknown"
                for _, _, type_uri in self.g.triples((o, RDF.type, None)):
                    entity_type = str(type_uri).split("/")[-1].split("#")[-1]
                    break
                
                label = str(o).split("/")[-1]
                for _, _, label_literal in self.g.triples((o, RDFS.label, None)):
                    label = str(label_literal)
                    break
                
                related_entities.append({
                    "uri": str(o),
                    "label": label,
                    "type": entity_type,
                    "distance": 1,
                    "path": [str(p)],
                    "direction": "outgoing"
                })
        
        # Incoming relations
        for s, p, o in self.g.triples((None, None, entity_uri)):
            if isinstance(s, URIRef) and s != entity_uri:
                # Get entity type and label
                entity_type = "Unknown"
                for _, _, type_uri in self.g.triples((s, RDF.type, None)):
                    entity_type = str(type_uri).split("/")[-1].split("#")[-1]
                    break
                
                label = str(s).split("/")[-1]
                for _, _, label_literal in self.g.triples((s, RDFS.label, None)):
                    label = str(label_literal)
                    break
                
                related_entities.append({
                    "uri": str(s),
                    "label": label,
                    "type": entity_type,
                    "distance": 1,
                    "path": [str(p)],
                    "direction": "incoming"
                })
        
        # TODO: Implement multi-hop traversal for distance > 1
        # This would require breadth-first search through the graph
        
        # Group by relation type
        relation_groups = {}
        for entity in related_entities:
            for path in entity.get("path", []):
                if path not in relation_groups:
                    relation_groups[path] = []
                relation_groups[path].append(entity)
        
        return {
            "related_entities": related_entities,
            "relation_groups": relation_groups,
            "entity_uri": str(entity_uri),
            "max_distance": max_distance
        }
    
    def run_sparql_query(self, query: str) -> Dict[str, Any]:
        """
        Run a SPARQL query on the RDF graph.
        
        Args:
            query: SPARQL query string
            
        Returns:
            Query results
        """
        try:
            logger.info(f"Running SPARQL query: {query[:100]}...")
            
            if self.use_rapids:
                # Try to use RAPIDS for the query first
                try:
                    # Convert to PropertyGraph
                    property_graph = self.convert_to_property_graph()
                    
                    # Run query with RAPIDS
                    if property_graph:
                        return self.rapids.run_sparql_query(property_graph, query)
                except Exception as e:
                    logger.warning(f"Error running SPARQL query with RAPIDS: {e}")
                    # Fall back to RDFlib
            
            # Use RDFlib for SPARQL queries
            if query.lower().startswith("select"):
                results = self.g.query(query)
                result_list = []
                
                # Convert results to dictionaries
                for row in results:
                    row_dict = {}
                    for i, var in enumerate(results.vars):
                        value = row[i]
                        
                        # Convert URIRef, Literal, etc. to string representation
                        if isinstance(value, URIRef):
                            row_dict[var] = str(value)
                        elif isinstance(value, Literal):
                            row_dict[var] = value.toPython()
                        elif isinstance(value, BNode):
                            row_dict[var] = f"_:{value}"
                        else:
                            row_dict[var] = str(value) if value is not None else None
                    
                    result_list.append(row_dict)
                
                return {
                    "query_type": "SELECT",
                    "results": result_list,
                    "vars": [str(var) for var in results.vars]
                }
                
            elif query.lower().startswith("construct"):
                result_graph = self.g.query(query).graph
                
                # Serialize the result graph to Turtle
                turtle = result_graph.serialize(format="turtle")
                
                return {
                    "query_type": "CONSTRUCT",
                    "results": turtle,
                    "triple_count": len(result_graph)
                }
                
            elif query.lower().startswith("ask"):
                result = bool(self.g.query(query).askAnswer)
                
                return {
                    "query_type": "ASK",
                    "result": result
                }
                
            else:
                # Other query types
                return {
                    "query_type": "UNKNOWN",
                    "message": "Unsupported query type",
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}", exc_info=True)
            return {
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def to_rdf(self, format: str = "turtle") -> str:
        """
        Serialize the graph to the specified RDF format.
        
        Args:
            format: Output format (turtle, xml, json-ld, n3, ntriples)
            
        Returns:
            RDF graph serialized in the specified format
        """
        format_map = {
            "turtle": "turtle",
            "ttl": "turtle",
            "xml": "xml",
            "rdf": "xml",
            "rdfxml": "xml",
            "json-ld": "json-ld",
            "jsonld": "json-ld",
            "n3": "n3",
            "ntriples": "nt",
            "nt": "nt"
        }
        
        # Get the actual format for rdflib
        rdf_format = format_map.get(format.lower(), "turtle")
        
        # Serialize the graph
        try:
            return self.g.serialize(format=rdf_format)
        except Exception as e:
            logger.error(f"Error serializing graph to {format}: {e}", exc_info=True)
            # Fallback to turtle
            return self.g.serialize(format="turtle")
    
    def save(self, file_path: str, format: str = "turtle") -> None:
        """
        Save the graph to a file.
        
        Args:
            file_path: Path to save the file
            format: Output format (turtle, xml, json-ld, n3, ntriples)
        """
        format_map = {
            "turtle": "turtle",
            "ttl": "turtle",
            "xml": "xml",
            "rdf": "xml",
            "rdfxml": "xml",
            "json-ld": "json-ld",
            "jsonld": "json-ld",
            "n3": "n3",
            "ntriples": "nt",
            "nt": "nt"
        }
        
        # Get the actual format for rdflib
        rdf_format = format_map.get(format.lower(), "turtle")
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Serialize the graph to the file
        try:
            self.g.serialize(destination=file_path, format=rdf_format)
            logger.info(f"Saved graph to {file_path} in {format} format")
        except Exception as e:
            logger.error(f"Error saving graph to {file_path}: {e}", exc_info=True)
            raise
    
    def compute_graph_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics for the knowledge graph.
        
        Returns:
            Dictionary of graph metrics
        """
        if self.use_rapids:
            # Try to use RAPIDS for computing metrics
            try:
                property_graph = self.convert_to_property_graph()
                if property_graph:
                    return self.rapids.compute_graph_metrics(property_graph)
            except Exception as e:
                logger.warning(f"Error computing graph metrics with RAPIDS: {e}")
                # Fall back to RDFlib
        
        # Use RDFlib for basic metrics
        metrics = {
            "triple_count": len(self.g),
            "subject_count": len(set(self.g.subjects())),
            "predicate_count": len(set(self.g.predicates())),
            "object_count": len(set(self.g.objects())),
        }
        
        # Count by type
        type_counts = {}
        for s, _, o in self.g.triples((None, RDF.type, None)):
            type_name = str(o).split("/")[-1].split("#")[-1]
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        metrics["type_counts"] = type_counts
        
        return metrics
    
    def to_turtle(self) -> str:
        """
        Serialize the graph to Turtle format.
        
        Returns:
            RDF graph serialized in Turtle format
        """
        return self.to_rdf(format="turtle")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up RAPIDS resources
        if self.rapids:
            self.rapids.cleanup()
        
        # Clean up Owlready2 resources
        if self.owlready2_available and self.world:
            try:
                self.world.close()
            except Exception as e:
                logger.warning(f"Error closing Owlready2 world: {e}")
    
    def clear(self) -> None:
        """Clear the graph for a new conversion."""
        self.g = Graph()
        self.triples = []
        self._init_namespaces()
    
    def __del__(self):
        """Cleanup when the object is garbage collected."""
        self.cleanup()
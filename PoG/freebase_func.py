from SPARQLWrapper import SPARQLWrapper, JSON, XML
# SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md
SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md


sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


sparql_tail_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?tailEntity
WHERE {
    ns:%s ?relation ?tailEntity .
}
"""

sparql_head_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?headEntity
WHERE {
    ?headEntity ?relation ns:%s .
}
"""


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def format(entity_id):
    return entity_id

def format1(entity_id):
    if "http://" in entity_id:
        return f"<{entity_id}>"
    else:
        return f"ns:{entity_id}"
    return entity_id

import time
from urllib.error import HTTPError

def execurte_sparql(sparql_txt):
    # Assuming SPARQLPATH is a variable that holds your SPARQL endpoint URL
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    print("Failed to execute after multiple attempts.")
    return None

# Sends a SPARQL query to a Freebase/Wikidata endpoint and retrieves structured data in JSON format.
#   Returns a list of dictionaries (extracted from "bindings") or None if the query fails.
def execute_sparql(sparql_txt):

    # Assuming SPARQLPATH is a variable that holds your SPARQL endpoint URL
    sparql = SPARQLWrapper(SPARQLPATH)      # Creates a SPARQL query wrapper
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)

            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    print("Failed to execute after multiple attempts.")
    return None
    # eg. raw qury
    # {
    # "head": { "vars": ["relation", "connectedEntity", "connectedEntityName", "direction"] },
    # "results": {
    #     "bindings": [
    #     {
    #         "relation": { "value": "http://rdf.freebase.com/ns/shares_border_with" },
    #         "connectedEntity": { "value": "http://rdf.freebase.com/ns/m.0f8l9c" },
    #         "connectedEntityName": { "value": "France" },
    #         "direction": { "value": "tail" }
    #     },
    #     {
    #         "relation": { "value": "http://rdf.freebase.com/ns/located_in" },
    #         "connectedEntity": { "value": "http://rdf.freebase.com/ns/m.0g7qf" },
    #         "connectedEntityName": { "value": "Europe" },
    #         "direction": { "value": "tail" }
    #     }
    #     ]
    # }
    # }


def replace_relation_prefix(relations):
    if relations is None:
        return []  
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    if entities is None:
        return []  
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]



from functools import lru_cache
import re

# Retrieves the name or type of an entity given its entity ID by querying a SPARQL knowledge graph (KG) endpoint.
#   Uses an LRU (Least Recently Used) caches results for up to 1,024 different entity queries for fast retrieval.
#       eg. id2entity_name_or_type("m.02jx3")  # First call → Runs SPARQL query
#           id2entity_name_or_type("m.02jx3")  # Second call → Uses cached result
#   Executes a SPARQL query to get the human-readable name of an entity.
@lru_cache(maxsize=1024)
def id2entity_name_or_type(entity_id):
    init_id = entity_id

    # Formats the SPARQL query string, replacing placeholders with entity_id
    entity_id = sparql_id % (format(entity_id), format(entity_id))

    # prepare SPARQL query
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(entity_id)
    sparql.setReturnFormat(JSON)
    # results = sparql.query().convert()

    # send SPARQL query
    results = []
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            break
            # return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    if attempts == 3:
        print("Failed to execute after multiple attempts.")

    # Process query result
    if len(results["results"]["bindings"]) == 0:    # if no results are found, return "Unnamed Entity"
        return "Unnamed Entity"
    else:
        # Extract entity name
        #   First, filter to find results with 'xml:lang': 'en'
        english_results = [result['tailEntity']['value'] for result in results["results"]["bindings"] if result['tailEntity'].get('xml:lang') == 'en']
        if english_results:
            return english_results[0]  # Return the first English result

        #   If no English labels are found, checks for names that contain only letters and numbers (ignores symbols)
        alphanumeric_results = [result['tailEntity']['value'] for result in results["results"]["bindings"]
                                if re.match("^[a-zA-Z0-9 ]+$", result['tailEntity']['value'])]
        if alphanumeric_results:
            return alphanumeric_results[0]  # Return the first alphanumeric result
        
        # If no English or alphanumeric names are found, returns "Unnamed Entity".
        return "Unnamed Entity"

# eg. with english result
# {
#   "results": {
#     "bindings": [
#       { "tailEntity": { "value": "Germany", "xml:lang": "en" } },
#       { "tailEntity": { "value": "Deutschland", "xml:lang": "de" } }
#     ]
#   }
# }
# Output: Germany

# eg. without english result
# {
#   "results": {
#     "bindings": [
#       { "tailEntity": { "value": "東京", "xml:lang": "ja" } },
#       { "tailEntity": { "value": "Tokyo" } }
#     ]
#   }
# }

### KGNET Accepted at <a href="https://icde2023.ics.uci.edu/">ICDE-2023</a>
# KGNET - A GML-Enabled RDF Engine
<img alt="kgnet_architecture" src="docs/imgs/kgnet.png" width="500"/>


<div style="text-align: justify">
<p>This vision paper proposes KGNet, a graph machine learning (GML)
enabled RDF engine. KGNet extends existing RDF engines with GML as a service to automate the training of GML models on KGs.
In KGNet, we maintain from the metadata of trained models a transparent RDF graph associated with the target knowledge graph (KG) to enable SPARQL queries to use these models while querying the target KG. The development of KGNet poses research opportunities
in various areas spanning GML pipeline automation, GML-Enabled SPARQL query optimization, and KG sampling for task-oriented training. The paper discusses the KGNet potential in supporting GML-enabled queries in real KGs of different application domains.
KGNet automatically opts to train models based on a given budget for node classification, link prediction, and semantic entity similarity. Using KGNet KG sampling, we achieved up 62% memory reduction and 35% faster training, while achieving similar or better accuracy w.r.t using the full KG.</p></div>

## Technical Report
Our technical report is available [here](docs/KGNET_CIDR_2023_technicalReport.pdf)

## Installation
* Clone the `kgnet` repo 
* Create `kgnet` Conda environment (Python 3.8) and install pip requirements.
* Activate the `kgnet` environment
```commandline
conda activate kgnet
```

## Quickstart
<ul>
<li>
<b>Install <a href="https://github.com/openlink/virtuoso-opensource">openlink Virtuoso Version 07.20.3229 </a> and load the knowledge graphs used in this paper. We use DBLP version <a href="https://dblp.org/rdf/release/">2022-06-01</a>, and <a href="https://data.deepai.org/FB15K.zip">FB15K</a>. </b>
</li>
<li>
<b> Prepaare endpoint URI for each graph to be used with kgnet. </b>
</li>
</ul>

<b>Generating the sampled graph dataset:</b>
1. under dataSampling the node sampler and edge sampler methods are used to sample large datasets for node classification and link prediction respectively. the sampling methods take the sull graph as input and the traget node or target edge and generate the task sampled subgraph
```python
# run node sampler 
python nodesampler.py --KG DBLP --targetNode https://dblp.org/rdf/schema-publishedIn
python edgesampler.py --KG FB15K --targetEdge /people/person/profession
```
2. Run data transformer [GML dataTransformer](/DataTransform/kgnet_data_transformer.py)
transform your dataset into adjaceny metrics for node classification task by providing 
   1. the triples (CSV/TSV) dataset 
   2. the splitting criterria i.e. https://dblp.org/rdf/schema#yearOfEvent
   3. the target nodes labels i.e. https://dblp.org/rdf/schema#publishedIn
The generated splits are
```
knoweldge graph dataset /
├── mapping
│   └── nodex_entid2name.csv
│   └── ....
├── raw
│   └── node-label
│   └── node-feat
│   └── relations
│      └── nodex_rel_nodey
│      └── ....
├── split
│   └── train
│   └── test
│   └── valid
...
```
```python
# run data transformer sampler 
python kgnet_data_transformer.py --KG DBLP --splittingEdge  https://dblp.org/rdf/schema-yearOfEvent --labelsEdge https://dblp.org/rdf/schema-publishedIn
```

3. Run your GML-Queries [GML Queries](/GMLOperators)
     - GML **Insert (Model Train)** Query
       ```python
       gml_query=""" prefix dblp:<https://www.dblp.org/>
       prefix kgnet:<https://www.kgnet.com/>
       Insert into <kgnet>  { ?s ?p ?o }
       where {select * from  kgnet.TrainGML(
       {modelName: 'MAG_Paper-Venue_Classifer',
        GML-Operator: kgnet:NodeClassifier,
        TargetNodes: dblp:publication,
        NodesLables: dblp:venue,
        aggregator: 'mean',
        activationFunction: 'sigmoid',
        hyperParameters:{ batchSize: 50,
                       n-layers:3,
                       h-dim:100 },
        budget:{ MaxMemory:50GB,
                MaxTime:1h,
                Priority:ModelScore} } )}""" 
   
        python InsertOperator.py --query gml_query
       ```
     - GML **Node Classifier** Query   
        ```python
        gml_query="""prefix dblp: <https://www.dblp.org/>
        prefix kgnet: <https://www.kgnet.com/>
        select ?title ?venue
        where { 
        ?paper a dblp:Publication.
        ?paper dblp:title ?title.
        ?paper ?NodeClassifier ?venue.
        ?NodeClassifier a kgnet:NodeClassifier.
        ?NodeClassifier kgnet:classifierTarget dblp:paper.
        ?NodeClassifier kgnet:classifierLabel dblp:venue.}"""   
      
         python NodeClassifier.py --query gml_query
         ```
     - GML **Delete** Query   
        ```python
        gml_query="""prefix dblp:<https://www.dblp.org/>
        prefix kgnet:<https://www.kgnet.com/>
        delete {?NodeClassifier ?p ?o}
        where {
        ?NodeClassifier a kgnet:NodeClassifier.
        ?NodeClassifier kgnet:classifierTarget dblp:paper.
        ?NodeClassifier kgnet:classifierLabel dblp:venue.}"""
      
        python InsertOperator.py --query gml_query
        ```
     - GML **Link Predictor** Query   
        ```python
         gml_query="""prefix fb: <https://www.freebase.com/>
         prefix kgnet: <https://www.kgnet.com/>
         select ?person ?profession
         where { ?person a fb:person.
         ?person ?LinkPredictor ?profession.
         ?LinkPredictor a kgnet:LinkPredictor.
         ?LinkPredictor kgnet:SourceNode fb:person.
         ?LinkPredictor kgnet:DestinationNode fb:Profession.
         ?LinkPredictor kgnet:TopK-Links 10.}"""  
       
         python LinkPredictor.py --query gml_query
         ```
     - GML **Similar Entity** Query
        ```python
           gml_query="""prefix dblp: <https://www.dblp.org/>
           prefix kgnet: <https://www.kgnet.com/>
           select ?paper ?similarPaper
           where { ?paper a dblp:publication.
           ?paper dblp:title ?title.
           ?paper ?Similar-Entity ?similarPaper.
           ?Similar-Entity a kgnet:Similar-Entity.
           ?Similar-Entity kgnet:TrainedFor dblp:paper.
           ?Similar-Entity kgnet:GRL_Model kgnet:KGE/ComplEx.
           ?Similar-Entity kgnet:Metric kgnet:CosineSim.}"""  
      
           python EntitySimilarity.py --query gml_query
         ```
  
##  Using the Kgnet Web Interface 
Kgnet provides predefined operators in form of python apis that allow seamless integration with a conventional data science pipeline.
Checkout our [rep](https://github.com/hussien/KGNet-Interface) and [KGNET APIs](GMLWebServiceApis)

## kgnet APIs
See the full list of supported GML-Operators [here](docs/kgnet_gml_operators.md).

## Citing Our Work
If you find our work useful, please cite it in your research:

## Publicity
This repository is part of our submission to CIDR-2023. We will make it available to the public research community upon acceptance. 

## Questions
For any questions please contact us at:<br/> hussein.abdallah@concordia.ca , essam.mansour@concordia.ca  

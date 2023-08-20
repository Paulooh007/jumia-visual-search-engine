<p  align="center">

<img  src="jumia_lens.png"  alt="project banner"  height=300  width=820/>

</p>  

# jumia-visual-search-engine ([Try out !!](https://huggingface.co/spaces/paulokewunmi/jumia_product_search))
==============================

A visual search engine for Jumia that lets users search for products by uploading an image. It uses computer vision to find similar or identical products within the store's inventory, saving users time and providing a more personalized shopping experience.

<p  align="center">

<img  src="demo.png"  alt="demo img"/>

</p>  

<details>
<summary>Click to expand/collapse</summary>

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── image_search_engine                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes image_search_engine a Python module
    │   ├── product_image_search.py    <- entry point for using `image_search_engine`
    │   │
    │   ├── artifacts           <- Submodule containing model artifacts and saved checkpoints.
    │   │
    │   ├── data       <- Submodule responsible for data management and data class definitions.
    │   │   ├── base_data_module.py
    │   │   └── jumia_3650_dataset.py
    │   │
    │   ├── metadata         <- Submodule containing metadata related to the data classes. 
    |   |
    │   ├── models         <- Submodule housing various PyTorch model classes for training/experimentation.       
    │   │   ├── base.py
    │   │   ├── arc_margin_product.py
    │   │   ├── efficientnet_ns.py
    │   │   └── gem_pooling.py
    |   |
    │   └── tests         <- Submodule dedicated to test scripts and sample data/images for model testing and validation.  
    │    
    ├── training           <- Includes scripts for train/experimentation and staging of models.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

</details>

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

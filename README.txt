ontoBuilder - An Ontology Builder ML playground

The Purpose of this project is to have a playground for experimenting with
ML techniques, trained on a series of PDF files.  Eventually these tools
will be used to train models on 70,000+ PDF files of data - for now, we
build up the tooling with just a limited set of random PDF data.

Note:

	** Assumes you have a bunch of PDF files in ~/Documents/PDF
	** Alter the path in ontoBuilderML/cp_random.sh if that is not the case.

Setup:

	Use the setup_python_environment.sh script to get the appropriate
	python-based tools installed in your system.

	$ ./setup_python_environment.sh

	** Always remember to "source ontoBuilderML/venv/bin/activate" to 
	activate the python3.11-based environment that is used in this
	project.
	
In the ontoBuilderML directory:

	$ cd ontoBuilderML/

	$ source venv/bin/activate  # local python3.11-based virtual env

	$ make prepare 
	# Copies 15 random files from ~/Documents/PDF to the pdfs/ directory

	$ make
	# Proceeds to unpack the PDF data, tokenize, train, etc.

	$ make clean
	# This will eradicate all data and 'clear' the ML completely
	# Remember: 'make prepare' will need to be done again!

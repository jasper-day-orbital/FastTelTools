# make clean
# make doc
# make test
# make venv
#
# Required dependencies
# - doc: doxygen
# - venv: virtualenv

doc:
	doxygen doxygen.config

venv:
	virtualenv venv --python=python3
	source venv/bin/activate && pip install -r requirements.txt --upgrade

test:
	pytest tests -v

# .cache (generated by pytest)
clean:
	rm -rf doc venv .cache

.PHONY: doc test venv

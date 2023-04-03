################################################################################################################################
#####     FILE COPIED WITH PERMISSION from: https://github.com/sugatoray/genespeak/blob/master/Makefile  Thanks to genespeak ###
################################################################################################################################
.PHONY: black flake test types install interrogate \
		build buildcheck buildplus buildcheckplus getpackageinfo \
		github pypi testpypi archive archive-spec \
		clean cleanall style check pipinstalltest \
		streamlit_demo streamlit_run freeup_port \
		this thispy thatpy


PACKAGE_NAME := lazytransform
TESTPYPI_DOWNLOAD_URL := "https://test.pypi.org/simple/"
PYPIPINSTALL := "python -m pip install -U --index-url"
PIPINSTALL_PYPITEST := "$(PYPIPINSTALL) $(TESTPYPI_DOWNLOAD_URL)"
PKG_INFO := "import pkginfo; dev = pkginfo.Develop('.'); print((dev.$${FIELD}))"
STREAMLIT_DEMO_APP := "./apps/demo/streamlit_app/app.py"
STREAMLIT_PORT := 12321
ARCHIVES_DIR := ".archives"

## Code maintenance

black:
	black --target-version py38 $(PACKAGE_NAME) tests apps setup.py

flake:
	flake8 $(PACKAGE_NAME) tests apps setup.py

test:
	pytest tests

types:
	python -m $(PACKAGE_NAME) tests

## Install for development

install:
	python -m pip install -e ".[dev]"
	pre-commit install

## Install from test.pypi.org

pipinstalltest:
	@if [ $(VERSION) ]; then $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME)==$(VERSION); else $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME); fi;

## Run interrogate

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 tests
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 $(PACKAGE_NAME)

## Build and Release

build: clean
	python setup.py sdist
	python setup.py bdist_wheel \
		# --universal

buildcheck: cleanall build
	twine check dist/*

buildplus: build getpackageinfo

buildcheckplus: buildcheck getpackageinfo

getpackageinfo:
	$(eval PKG_NAME := $(shell FIELD="name" && python -c $(PKG_INFO);))
	@echo PKG_NAME is: [$(PKG_NAME)]
	$(eval PKG_VERSION := $(shell FIELD="version" && python -c $(PKG_INFO);))
	@echo PKG_VERSION is: [$(PKG_VERSION)]

github: buildplus
	# creating a github release: https://cli.github.com/manual/gh_release_create
	gh release create v$(PKG_VERSION)

githubplus: buildplus
	# creating a github release: https://cli.github.com/manual/gh_release_create
	gh release create v$(PKG_VERSION) ./dist/$(PKG_NAME)-$(PKG_VERSION)*.*

pypi: build
	twine upload dist/*

testpypi: build
	# source: https://packaging.python.org/en/latest/guides/using-testpypi/#using-testpypi-with-twine
	twine upload --repository testpypi dist/*

## Creating Git Archives (Similar to GitHub Releases)

archive-spec: buildplus
	@# Usage:
	@# $ make archive-spec
	@# $ make archive-spec REF="v0.0.9"
	@# $ make archive-spec REF="master"
	$(eval REF := $(shell if [ -z $(REF) ]; then echo HEAD; else echo $(REF); fi))
	@echo REF is: [$(REF)]
	$(eval ARCHIVE_FORMAT := "tar.gz")
	$(eval TAG := $(shell if [ "$(REF)" = "HEAD" ]; then echo v$(PKG_VERSION); else echo $(REF); fi))
	@echo TAG is: [$(TAG)]
	$(eval ARCHIVE_FILEPATH := $(ARCHIVES_DIR)/$(TAG).$(ARCHIVE_FORMAT))

archive-usage:
	#-----------------------------
	# Usage:
	# $ make archive
	# $ make archive REF="v0.0.9"
	# $ make archive REF="master"
	#-----------------------------
	@echo ""

archive: archive-usage archive-spec
	@mkdir -p $(ARCHIVES_DIR)
	@echo "Creating git archive: [$(ARCHIVE_FILEPATH)]\n"
	@# Example: git archive --prefix=v0.0.9/ -o .archives/v0.0.9.tar.gz HEAD
	git archive --prefix=$(TAG)/ -o $(ARCHIVE_FILEPATH) $(REF)

## Cleaning up repository

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

cleanall: clean
	rm -rf build/* dist/* $(PACKAGE_NAME).egg-info/* $(ARCHIVES_DIR)/*

## Style Checks and Unit Tests

style: clean black flake interrogate clean

check: clean black flake interrogate test clean

## Streamlit App

streamlit_demo:
	# Note: To run the following command the port 8051 needs to be available.
	#       If somehow, a previously running streamlit session did not exit
	#		properly, you need to manually and forcibly stop it.
	#       Stop an already running streamlit server:
	#
	#       sudo fuser -k 8051/tcp
	streamlit run $(STREAMLIT_DEMO_APP) --server.port=8051 &

streamlit_run:
	# Note: To run the following command the port 8051 needs to be available.
	#       If somehow, a previously running streamlit session did not exit
	#		properly, you need to manually and forcibly stop it.
	#       Stop an already running streamlit server:
	#
	#       sudo fuser -k $(STREAMLIT_PORT)/tcp
	streamlit run $(STREAMLIT_DEMO_APP) --server.port=$(STREAMLIT_PORT) &

## Release a Port

freeup_port:
	#-----------------------------
	# Note:
	# 		Default PORT=8051
	# Usage:
	# $ freeup_port PORT=8051
	#
	#-----------------------------
	$(eval PORT := $(shell if [ -z $(PORT) ]; then echo 8051; else echo $(PORT); fi))
	sudo fuser -k $(PORT)/tcp

################  BELOW THIS LINE:  MEANT FOR TESTING ONLY    ################

this:
	# example: make this VERSION="0.0.3"
	@if [ $(VERSION) ]; then echo This is $(PACKAGE_NAME)==$(VERSION); else echo This is $(PACKAGE_NAME); fi;

thispy:
	#  example: https://lists.gnu.org/archive/html/help-make/2015-03/msg00011.html
	@FIELD="name" && python -c $(PKG_INFO);
	@FIELD="version" && python -c $(PKG_INFO);
	$(eval FIELD := "name")
	@echo FIELD is: [$(FIELD)]

thatpy: thispy
	@echo FIELD is: [$(FIELD)]
	$(eval BRANCH := $(shell if [ -z $(BRANCH) ]; then echo HEAD; else echo $(BRANCH); fi))
	@echo BRANCH is: [$(BRANCH)]

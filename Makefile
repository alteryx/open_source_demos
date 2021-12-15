.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='featuretools|composeml|evalml|woodwork|bokeh')
	pip freeze | grep -v "open_source_demos.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

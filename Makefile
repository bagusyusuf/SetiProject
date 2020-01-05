all: report.pdf

.PHONY: report.pdf
report.pdf:
	pandoc report.md -o report.pdf

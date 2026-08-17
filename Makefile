# posit-scaling -- reproduce everything with `make all`
PY := python3
EXP := experiments

.PHONY: all test theory represent sweep decision figures paper clean help

help:
	@echo "make test       validate the quantizer against all 65,536 posit16 values"
	@echo "make theory     Result 2: LP optimum vs Karp max mean cycle   (~10 s)"
	@echo "make represent  Results 1 and 3: representation fidelity      (~20 s)"
	@echo "make decision   decision rule: removability vs which scaling  (~15 s)"
	@echo "make sweep      Result 4: 1,728 LU solves                     (~70 s)"
	@echo "make figures    regenerate all figures from results/"
	@echo "make paper      build paper/main.pdf (needs pdflatex)"
	@echo "make all        everything above except paper"
	@echo "make clean      remove generated CSVs and LaTeX artifacts"

test:
	$(PY) -m pytest tests/ -q

theory:
	cd $(EXP) && $(PY) run_theory_check.py

represent:
	cd $(EXP) && $(PY) run_representation.py

decision:
	cd $(EXP) && $(PY) run_decision_rule.py

sweep:
	cd $(EXP) && $(PY) run_sweep.py

figures:
	cd $(EXP) && $(PY) make_figures.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && echo "paper/main.pdf built"

all: test theory represent decision sweep figures
	@echo ""
	@echo "All experiments reproduced. Figures in paper/figures/."

clean:
	rm -f results/*.csv
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.synctex.gz
	find . -name __pycache__ -type d -exec rm -rf {} +

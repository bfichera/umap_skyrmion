.PHONY: clean

main: fig1_src/.done

fig1_src/.done: summary_plots.py
	mkdir -p fig1_src && \
	../venv/bin/python summary_plots.py ../replicate_umap/results/128_002.pkl fig1_src --num_frames 6 --ttcfs_i 0 0 0 0 0 6 6 6 6 6 12 12 12 12 12 18 18 18 18 18 24 24 24 24 24 --ttcfs_j 0 6 12 18 24 0 6 12 18 24 0 6 12 18 24 0 6 12 18 24 0 6 12 18 24 && \
	touch $@

clean:
	rm -r fig1_src

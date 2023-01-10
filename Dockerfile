FROM inseefrlab/onyxia-vscode-python:py3.10.4

RUN git clone https://github.com/ThomasFaria/DT-RN-chapitre4.git && \
    cd DT-RN-chapitre4 && \
    pip install -r requirements.txt && \
    chown -R ${USERNAME}:${GROUPNAME} ${HOME}

FROM nvcr.io/nvidia/jax:26.08-py3

# feax revision to build against (branch, tag or commit of github.com/Naruki-Ichihara/feax)
ARG FEAX_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Pin the base image's JAX / jaxlib. The NVCR image ships JAX built against its
# CUDA plugins; a PyPI jax/jaxlib pulled in by a dependency breaks the plugin ABI.
# PIP_CONSTRAINT also applies to pip run later inside the container.
RUN python -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('jax', 'jaxlib')))" \
        > /etc/pip-constraints-jax.txt && \
    cat /etc/pip-constraints-jax.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints-jax.txt

# feax from source at FEAX_REF: its core + cuda13 dependencies from its
# pyproject.toml, then feax itself with --no-deps. asdex is installed with
# --no-deps, as in feax's own Dockerfile: it declares jax>=0.11.0, which the NVCR
# pre-release build (0.11.0.devYYYYMMDD) does not satisfy under PEP 440.
RUN git clone --depth 1 --branch ${FEAX_REF} https://github.com/Naruki-Ichihara/feax.git /opt/feax || \
    (git clone https://github.com/Naruki-Ichihara/feax.git /opt/feax && git -C /opt/feax checkout ${FEAX_REF})
RUN cd /opt/feax && \
    python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; deps = p['dependencies'] + p['optional-dependencies']['cuda13']; print('\n'.join(d for d in deps if not d.startswith('asdex')))" \
        > /tmp/feax-requirements.txt && \
    pip install -r /tmp/feax-requirements.txt && \
    pip install --no-deps asdex && \
    pip install --no-deps /opt/feax

# spineax (cuDSS 1.x direct solver used by feax) from the fork, built against the
# container's jaxlib; --no-deps keeps the base JAX (see feax's Dockerfile).
RUN pip install "nvidia-cudss-cu13>=1.0.0,<2" nvidia-cusparse nvidia-cuda-cccl && \
    pip install --no-deps lineax && \
    pip install --no-build-isolation "scikit-build-core>=0.5" nanobind && \
    pip install --no-build-isolation --no-deps \
        "git+https://github.com/Naruki-Ichihara/spineax.git"

# lattice: its dependencies except feax (installed above), then the package
# itself, editable: docker-compose mounts the source tree over /workspace.
COPY . /workspace
WORKDIR /workspace
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; deps = p['dependencies'] + p['optional-dependencies']['plot'] + p['optional-dependencies']['dev']; print('\n'.join(d for d in deps if not d.startswith('feax')))" \
        > /tmp/lattice-requirements.txt && \
    pip install -r /tmp/lattice-requirements.txt && \
    pip install --no-deps -e .

# Fail the build if the base JAX was replaced or the stack does not import.
RUN python -c "import importlib.metadata as m; pins = dict(l.strip().split('==') for l in open('/etc/pip-constraints-jax.txt')); bad = {p: m.version(p) for p, v in pins.items() if m.version(p) != v}; assert not bad, f'base JAX replaced: {bad} (expected {pins})'; import feax, exponax, lattice"

CMD ["/bin/bash"]

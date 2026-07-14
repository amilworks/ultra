# syntax=docker/dockerfile:1.7

# This is a reviewed reconstruction variant, not an upstream-published image.
# Upstream commit 1803a6ab uses Python 3.11.8 in its Dockerfile, while its
# exported requirements activate only on Python >=3.13. Python 3.13 cannot
# install the locked NumPy 1.26.4 binary. Keep the upstream interpreter and
# make only the conflicting global lower-bound marker executable on 3.11.
ARG PYTHON_BASE_IMAGE=python:3.11.8@sha256:61d662f6d52206ab2290af4258257b5369573b6a4bbd904896699cc909221334
FROM ${PYTHON_BASE_IMAGE}

ARG PYTHON_BASE_IMAGE
ARG TARGETPLATFORM
ARG MATTOOLS_REVISION
ARG MATTOOLS_MANIFEST_SHA256
ARG UPSTREAM_REQUIREMENTS_SHA256
ARG ADAPTED_REQUIREMENTS_SHA256
ARG SUPPLEMENTAL_REQUIREMENTS_SHA256
ARG TOOL_SOURCE_MANIFEST_SHA256
ARG CANDIDATE_FIXTURE_FILE_COUNT
ARG CANDIDATE_FIXTURE_MANIFEST_SHA256
ARG SAFE_PARSER_SHA256
ARG RUNNER_WRAPPER_SHA256
ARG STRICT_SHADOW_SHA256
ARG SEMANTIC_REPAIRS_SHA256

LABEL org.opencontainers.image.title="MatTools evaluator reconstruction variant" \
      org.opencontainers.image.revision="${MATTOOLS_REVISION}" \
      org.opencontainers.image.description="Fail-closed Python 3.11 reconstruction; no official upstream image artifact is claimed" \
      io.ultra.mattools.official-artifact="false" \
      io.ultra.mattools.environment-kind="reviewed-reconstruction-variant" \
      io.ultra.mattools.base-image="${PYTHON_BASE_IMAGE}" \
      io.ultra.mattools.target-platform="${TARGETPLATFORM}" \
      io.ultra.mattools.snapshot-manifest-sha256="${MATTOOLS_MANIFEST_SHA256}" \
      io.ultra.mattools.upstream-requirements-sha256="${UPSTREAM_REQUIREMENTS_SHA256}" \
      io.ultra.mattools.adapted-requirements-sha256="${ADAPTED_REQUIREMENTS_SHA256}" \
      io.ultra.mattools.supplemental-requirements-sha256="${SUPPLEMENTAL_REQUIREMENTS_SHA256}" \
      io.ultra.mattools.tool-source-manifest-sha256="${TOOL_SOURCE_MANIFEST_SHA256}" \
      io.ultra.mattools.candidate-fixture-file-count="${CANDIDATE_FIXTURE_FILE_COUNT}" \
      io.ultra.mattools.candidate-fixture-manifest-sha256="${CANDIDATE_FIXTURE_MANIFEST_SHA256}" \
      io.ultra.mattools.candidate-visible-source-policy="input-fixtures-only" \
      io.ultra.mattools.safe-parser-sha256="${SAFE_PARSER_SHA256}" \
      io.ultra.mattools.runner-wrapper-sha256="${RUNNER_WRAPPER_SHA256}" \
      io.ultra.mattools.strict-shadow-sha256="${STRICT_SHADOW_SHA256}" \
      io.ultra.mattools.semantic-repairs-sha256="${SEMANTIC_REPAIRS_SHA256}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=mattools requirements.txt /app/upstream-requirements.txt
COPY --from=mattools src/tool_source_code/pymatgen-analysis-defects/tests/test_files /app/tool_source_code/pymatgen-analysis-defects/tests/test_files
COPY deploy/docker/mattools-evaluator-supplemental-requirements.txt /app/supplemental-requirements.txt

RUN test "$(sha256sum /app/upstream-requirements.txt | cut -d ' ' -f 1)" = "${UPSTREAM_REQUIREMENTS_SHA256}" \
    && test "$(sha256sum /app/supplemental-requirements.txt | cut -d ' ' -f 1)" = "${SUPPLEMENTAL_REQUIREMENTS_SHA256}" \
    && sed -e 's/python_version >= "3.13"/python_version >= "3.11"/g' \
        -e 's/python_version == "3.13"/python_version == "3.11"/g' \
        /app/upstream-requirements.txt > /app/evaluator-requirements.txt \
    && test "$(sha256sum /app/evaluator-requirements.txt | cut -d ' ' -f 1)" = "${ADAPTED_REQUIREMENTS_SHA256}"

RUN python -m pip install --require-hashes \
        -r /app/evaluator-requirements.txt \
        -r /app/supplemental-requirements.txt \
    && python -m pip check \
    && python -c "from importlib.metadata import version; assert version('pymatgen') == '2024.8.9'; assert version('pymatgen-analysis-defects') == '2024.7.19'"

CMD ["bash"]

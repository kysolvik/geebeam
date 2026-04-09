FROM apache/beam_python3.11_sdk:2.72.0

# Get uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install package
RUN uv pip install --system "git+https://github.com/kysolvik/geebeam@main#egg=geebeam[tensorflow]"

RUN uv pip check --system

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]

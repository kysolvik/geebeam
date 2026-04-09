FROM apache/beam_python3.11_sdk:2.72.0

# Install package
RUN pip install "git+https://github.com/kysolvik/geebeam@main[tensorflow]"

RUN pip check

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]

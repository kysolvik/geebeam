FROM apache/beam_python3.10_sdk:2.71.0

# Install package
RUN pip install git+https://github.com/kysolvik/geebeam@main

RUN pip check

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]

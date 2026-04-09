FROM apache/beam_python3.11_sdk:2.71.0

# Install package
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir "git+https://github.com/kysolvik/geebeam@main#egg=geebeam[tensorflow]"

RUN pip check

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]

FROM dceoy/pdbio:latest

ADD . /tmp/vgmmfilter

RUN set -e \
      && apt-get -y update \
      && apt-get -y dist-upgrade \
      && apt-get -y autoremove \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

RUN set -e \
      && pip install -U --no-cache-dir pip /tmp/vgmmfilter \
      && rm -rf /tmp/vgmmfilter

ENTRYPOINT ["jupyter"]

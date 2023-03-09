# This file is a template, and might need editing before it works on your project.
# This Dockerfile installs a compiled binary into a bare system.
# You must either commit your compiled binary into source control (not recommended)
# or build the binary first as part of a CI/CD pipeline.

FROM alpine:3.14.2
MAINTAINER adambaj@seznam.cz
LABEL org.label-schema.name = "Alpine C/C++ build environment with CMake and Boost"
LABEL org.label-schema.description = "Want to have a CI/CD pipeline for a C++ with standard 23? Wanna have valgrind?"

ARG BOOST_VERSION=1.77.0
ARG BOOST_DIR=boost_1_77_0
ENV BOOST_VERSION ${BOOST_VERSION}

# We'll likely need to add SSL root certificates
RUN apk -U --no-cache add ca-certificates g++ 'cmake>3.20.2' libstdc++ build-base valgrind && \
    apk -U upgrade && \
    update-ca-certificates

# Install boost based on above setting
RUN apk add --no-cache --virtual .build-dependencies \
    openssl \
    linux-headers \
    # https://sourceforge.net/projects/boost/files/boost/1.77.0/boost_1_77_0.tar.bz2/download
    && wget -O ${BOOST_DIR}.tar.bz2 https://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/${BOOST_DIR}.tar.bz2/download\
    && tar --bzip2 -xf ${BOOST_DIR}.tar.bz2 \
    && cd ${BOOST_DIR} \
    && ./bootstrap.sh \
    && ./b2 --without-python --prefix=/usr -j 4 link=shared runtime-link=shared install \
    && cd .. && rm -rf ${BOOST_DIR} ${BOOST_DIR}.tar.bz2 \
    && apk del .build-dependencies


WORKDIR /usr/local/bin
#ADD tmp.sh tmp.sh



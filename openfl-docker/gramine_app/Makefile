# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
VENV_ROOT ?= /opt/venv
WORKSPACE_ROOT ?= /workspace
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: fx.manifest
ifeq ($(SGX),1)
all: fx.manifest.sgx fx.sig
endif

fx.manifest: fx.manifest.template
	@echo "Making fx.manifest file"
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dvenv_root=$(VENV_ROOT) \
		-Dentrypoint=$(VENV_ROOT)/bin/fx \
		-Dworkspace_root=$(WORKSPACE_ROOT) \
		$< >$@

fx.manifest.sgx: fx.manifest
	@echo "Making fx.manifest.sgx file"
	@test -s $(SGX_SIGNER_KEY) || \
		{ echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	@gramine-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--manifest $< \
		--output $@ | tail -n 1 | tr -d ' ' | xargs -I {} echo "fx.mr_enclave={}"

fx.sig: fx.manifest.sgx

.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT* *.PID TEST_STDOUT TEST_STDERR
	$(RM) -r scripts/__pycache__

.PHONY: distclean
distclean: clean

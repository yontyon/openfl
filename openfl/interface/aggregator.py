# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregator module."""

import sys
from logging import getLogger
from pathlib import Path

from click import (
    Path as ClickPath,
)
from click import (
    confirm,
    echo,
    group,
    option,
    pass_context,
    style,
)

from openfl.cryptography.ca import sign_certificate
from openfl.cryptography.io import (
    get_csr_hash,
    read_crt,
    read_csr,
    read_key,
    write_crt,
    write_key,
)
from openfl.cryptography.participant import generate_csr
from openfl.federated import Plan
from openfl.interface.cli_helper import CERT_DIR
from openfl.utilities import click_types
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.utils import getfqdn_env

logger = getLogger(__name__)


@group()
@pass_context
def aggregator(context):
    """
    Manage Federated Learning Aggregator.

    Args:
        context (click.Context): The context passed from the CLI.
    """
    context.obj["group"] = "aggregator"


@aggregator.command(name="start")
@option(
    "-p",
    "--plan",
    required=False,
    help="Federated learning plan [plan/plan.yaml]",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-c",
    "--authorized_cols",
    required=False,
    help="Authorized collaborator list [plan/cols.yaml]",
    default="plan/cols.yaml",
    type=ClickPath(exists=True),
)
@option(
    "--task_group",
    required=False,
    default="learning",
    help="Selected task-group for assignment - defaults to learning",
)
def start_(plan, authorized_cols, task_group):
    """Start the aggregator service.

    Args:
        plan (str): Path to plan config file
        authorized_cols (str): Path to authorized collaborators file
        task_group (str): Selected task-group for assignement - defaults to 'learning'
    """
    if is_directory_traversal(plan):
        echo("Federated learning plan path is out of the openfl workspace scope.")
        sys.exit(1)
    if is_directory_traversal(authorized_cols):
        echo("Authorized collaborator list file path is out of the openfl workspace scope.")
        sys.exit(1)

    # Parse plan and override mode if specified
    parsed_plan = Plan.parse(
        plan_config_path=Path(plan).absolute(),
        cols_config_path=Path(authorized_cols).absolute(),
    )

    # Set task_group in aggregator settings
    if "settings" not in parsed_plan.config["aggregator"]:
        parsed_plan.config["aggregator"]["settings"] = {}
    parsed_plan.config["aggregator"]["settings"]["task_group"] = task_group
    logger.info(f"Setting aggregator to assign: {task_group} task_group")

    logger.info("🧿 Starting the Aggregator Service.")

    parsed_plan.get_server().serve()


@aggregator.command(name="generate-cert-request")
@option(
    "--fqdn",
    required=False,
    type=click_types.FQDN,
    help=f"The fully qualified domain name of aggregator node [{getfqdn_env()}]",
    default=getfqdn_env(),
)
def _generate_cert_request(fqdn):
    """Create aggregator certificate key pair.

    Args:
        fqdn (str): The fully qualified domain name of aggregator node.
    """
    generate_cert_request(fqdn)


def generate_cert_request(fqdn):
    """Create aggregator certificate key pair."""

    if fqdn is None:
        fqdn = getfqdn_env()

    common_name = f"{fqdn}".lower()
    subject_alternative_name = f"DNS:{common_name}"
    file_name = f"agg_{common_name}"

    echo(
        f"Creating AGGREGATOR certificate key pair with following settings: "
        f"CN={style(common_name, fg='red')},"
        f" SAN={style(subject_alternative_name, fg='red')}"
    )

    server_private_key, server_csr = generate_csr(common_name, server=True)

    (CERT_DIR / "server").mkdir(parents=True, exist_ok=True)

    echo("  Writing AGGREGATOR certificate key pair to: " + style(f"{CERT_DIR}/server", fg="green"))

    # Print csr hash before writing csr to disk
    csr_hash = get_csr_hash(server_csr)
    echo("The CSR Hash " + style(f"{csr_hash}", fg="red"))

    # Write aggregator csr and key to disk
    write_crt(server_csr, CERT_DIR / "server" / f"{file_name}.csr")
    write_key(server_private_key, CERT_DIR / "server" / f"{file_name}.key")


# TODO: function not used
def find_certificate_name(file_name):
    """Search the CRT for the actual aggregator name.

    Args:
        file_name (str): The name of the file to search.

    Returns:
        str: The name of the aggregator found in the CRT.
    """
    # This loop looks for the collaborator name in the key
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if "Subject: CN=" in line:
                col_name = line.split("=")[-1].strip()
                break
    return col_name


@aggregator.command(name="certify")
@option(
    "-n",
    "--fqdn",
    type=click_types.FQDN,
    help=f"The fully qualified domain name of aggregator node [{getfqdn_env()}]",
    default=getfqdn_env(),
)
@option("-s", "--silent", help="Do not prompt", is_flag=True)
def _certify(fqdn, silent):
    """Sign/certify the aggregator certificate key pair."""
    certify(fqdn, silent)


def certify(fqdn, silent):
    """Sign/certify the aggregator certificate key pair.

    Args:
        fqdn (str): The fully qualified domain name of aggregator node.
        silent (bool): Flag to enable silent mode.
    """

    if fqdn is None:
        fqdn = getfqdn_env()

    common_name = f"{fqdn}".lower()
    file_name = f"agg_{common_name}"
    cert_name = f"server/{file_name}"
    signing_key_path = "ca/signing-ca/private/signing-ca.key"
    signing_crt_path = "ca/signing-ca.crt"

    # Load CSR
    csr_path_absolute_path = Path(CERT_DIR / f"{cert_name}.csr").absolute()
    if not csr_path_absolute_path.exists():
        echo(
            style("Aggregator certificate signing request not found.", fg="red")
            + " Please run `fx aggregator generate-cert-request`"
            " to generate the certificate request."
        )

    csr, csr_hash = read_csr(csr_path_absolute_path)

    # Load private signing key
    private_sign_key_absolute_path = Path(CERT_DIR / signing_key_path).absolute()
    if not private_sign_key_absolute_path.exists():
        echo(
            style("Signing key not found.", fg="red") + " Please run `fx workspace certify`"
            " to initialize the local certificate authority."
        )

    signing_key = read_key(private_sign_key_absolute_path)

    # Load signing cert
    signing_crt_absolute_path = Path(CERT_DIR / signing_crt_path).absolute()
    if not signing_crt_absolute_path.exists():
        echo(
            style("Signing certificate not found.", fg="red") + " Please run `fx workspace certify`"
            " to initialize the local certificate authority."
        )

    signing_crt = read_crt(signing_crt_absolute_path)

    echo(
        "The CSR Hash for file "
        + style(f"{cert_name}.csr", fg="green")
        + " = "
        + style(f"{csr_hash}", fg="red")
    )

    crt_path_absolute_path = Path(CERT_DIR / f"{cert_name}.crt").absolute()

    if silent:
        echo(" Warning: manual check of certificate hashes is bypassed in silent mode.")
        echo(" Signing AGGREGATOR certificate")
        signed_agg_cert = sign_certificate(csr, signing_key, signing_crt.subject)
        write_crt(signed_agg_cert, crt_path_absolute_path)

    else:
        echo("Make sure the two hashes above are the same.")
        if confirm("Do you want to sign this certificate?"):
            echo(" Signing AGGREGATOR certificate")
            signed_agg_cert = sign_certificate(csr, signing_key, signing_crt.subject)
            write_crt(signed_agg_cert, crt_path_absolute_path)

        else:
            echo(
                style("Not signing certificate.", fg="red")
                + " Please check with this AGGREGATOR to get the correct"
                " certificate for this federation."
            )

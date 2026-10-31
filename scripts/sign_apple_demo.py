#!/usr/bin/env python3
"""Sign the demo using an existing local development profile. No portal/API changes."""
import argparse
import fnmatch
import hashlib
import pathlib
import plistlib
import re
import shutil
import subprocess
import tempfile


def sign(app, profile_path):
    profile = plistlib.loads(subprocess.check_output(["security", "cms", "-D", "-i", str(profile_path)]))
    allowed = profile.get("Entitlements", {})
    if not allowed.get("get-task-allow"):
        raise ValueError("Use a development provisioning profile, not a distribution profile")
    info = plistlib.loads((app / "Info.plist").read_bytes())
    team = profile["TeamIdentifier"][0]
    prefix = profile["ApplicationIdentifierPrefix"][0]
    app_id = prefix + "." + info["CFBundleIdentifier"]
    if not fnmatch.fnmatchcase(app_id, allowed["application-identifier"]):
        raise ValueError("The profile does not cover this demo's bundle identifier")
    fingerprints = {hashlib.sha1(cert).hexdigest().upper() for cert in profile["DeveloperCertificates"]}
    identities = subprocess.check_output(["security", "find-identity", "-v", "-p", "codesigning"], text=True)
    identity = next((key for key in re.findall(r"\b[0-9A-F]{40}\b", identities) if key in fingerprints), None)
    if identity is None:
        raise ValueError("No valid local signing identity matches the development profile")
    entitlements = {"application-identifier": app_id, "com.apple.developer.team-identifier": team,
                    "get-task-allow": True, "keychain-access-groups": [app_id]}
    shutil.copy2(profile_path, app / "embedded.mobileprovision")
    magic = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"}
    main = app / info["CFBundleExecutable"]
    for path in app.rglob("*"):
        if path.is_file() and path != main:
            with path.open("rb") as stream:
                executable = stream.read(4) in magic
            if executable:
                subprocess.run(["codesign", "--force", "--sign", identity, str(path)], check=True, capture_output=True)
    with tempfile.TemporaryDirectory(prefix="gliner-sign-") as temporary:
        entfile = pathlib.Path(temporary) / "entitlements.plist"
        entfile.write_bytes(plistlib.dumps(entitlements))
        subprocess.run(["codesign", "--force", "--sign", identity, "--entitlements", str(entfile), str(app)],
                       check=True, capture_output=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True, capture_output=True)
    print("Development signature verified. No App Store or provisioning-portal changes were made.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=pathlib.Path, default=pathlib.Path(".derivedData/Build/Products/Debug-iphoneos/GLiNERDemo.app"))
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sign(args.app.resolve(), args.profile.resolve())

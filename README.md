# flood-area-segmentation
Einteilung in Überschwemmungsgebiet (Wasserbereiche auf den Bildern zuverlässig erkennt)


# SSH to the server
1. Add this config to `.ssh/config`

```
Host aie-project
    Hostname 77.237.53.194
    Port 24
    User aie3
    PasswordAuthentication yes
```

2. Connect to the server via VSCode
F1 -> `Remote SSH: Connect current Window ...` -> Choose aie-project


# Install dependencies
```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```


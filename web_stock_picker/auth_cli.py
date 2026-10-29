#!/usr/bin/env python3
import argparse, getpass
import auth

p = argparse.ArgumentParser()
s = p.add_subparsers(dest="cmd", required=True)
c = s.add_parser("create-admin"); c.add_argument("username")
d = s.add_parser("delete-user"); d.add_argument("username")
s.add_parser("list")
a = p.parse_args()

if a.cmd == "create-admin":
    pw = getpass.getpass("新密码: ")
    if pw != getpass.getpass("再输一次: "): raise SystemExit("两次不一致")
    secret, uri, rec = auth.create_user(a.username, pw)
    print("TOTP secret:", secret); print("otpauth URI:", uri)
    print("恢复码（只显示一次，离线保存）:")
    for r in rec: print("  ", r)
elif a.cmd == "delete-user":
    print("已删除" if auth.delete_user(a.username) else "用户不存在")
elif a.cmd == "list":
    print("\n".join(auth.load_users().keys()) or "(空)")

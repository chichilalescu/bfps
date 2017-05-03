from bfps.DNS import DNS


def main():
    c = DNS()
    c.write_src()
    c.compile_code()
    return None

if __name__ == '__main__':
    main()


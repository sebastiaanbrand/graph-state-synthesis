
import gs_bmc_encoder

def main():
    with open('example/source.tgf', 'r', encoding='utf-8') as source:
         source_tgf = source.read()
    with open('example/target.tgf', 'r', encoding='utf-8') as target:
        target_tgf = target.read()
    num_nodes = 4
    depth = 2

    cnf = gs_bmc_encoder.encode_bmc(source_tgf, target_tgf, num_nodes, depth)
    print(cnf)

if __name__ == '__main__':
    main()

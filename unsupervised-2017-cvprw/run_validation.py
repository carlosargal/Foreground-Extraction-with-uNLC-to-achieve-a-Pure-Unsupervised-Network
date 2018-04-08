import mfb_pretrain_dis_val

def run_val():
    for i in range(100, 8500, 100):
        checkpoint = '/mfb_dis_uNLC_final_ucf24.model-%s' % i
        mfb_pretrain_dis_val.run_validating(checkpoint, i)

def main():
    run_val()

if __name__ == '__main__':
    main()
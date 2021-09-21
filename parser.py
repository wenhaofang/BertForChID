import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default = 'main', help = '')

    parser.add_argument('--train_input', default = 'data/chid/train_data.txt', help = 'path to train data')
    parser.add_argument('--valid_input', default = 'data/chid/dev_data.txt', help = 'path to valid data')
    parser.add_argument('--test_input' , default = 'data/chid/test_data.txt' , help = 'path to test data' )
    parser.add_argument('--model_name_or_path', default = 'data/bert', help = 'path to pre-trained bert model')

    parser.add_argument('--hidden_size', type = int, default = 768, help = '')
    parser.add_argument('--num_classes', type = int, default = 2, help = '')
    parser.add_argument('--num_choices', type = int, default = 7, help = '')

    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--max_seq_len', type = int, default = 128, help = '')
    parser.add_argument('--learning_rate', type = float, default = 1e-5, help = '')
    parser.add_argument('--n_epoch', type = int, default = 10, help = '')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_option()

import argparse

if __name__ == '__main__':
    default_config = {
        'batch_size': 1024,
        'sep': '|',
    }

    parser = argparse.ArgumentParser('brute force nearest neighbor search')
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'],
                        help=f'query batch size; default: {default_config["batch_size"]}')
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--sep', default=default_config['sep'],
                        help=f'separator for vector representation; default: {default_config["sep"]}')
    parser.add_argument('--dim', type=int, required=True, help='expected vector dimension; all non-compliant entry will be ignored')
    parser.add_argument('--backend', choices=['pytorch', 'faiss'], default='pytorch',
                        help='brute force search backend')
    parser.add_argument('--gpu', action='store_true', help='use gpu')

    args = parser.parse_args()
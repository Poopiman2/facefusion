import argparse
from facefusion import core
from facefusion import state_manager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-providers', type=str, default='cpu', help='Execution provider: cuda, cpu, etc.')
    args = parser.parse_args()

    # Save it to state_manager so core/other modules can access it
    state_manager.set_item('execution_provider', args.execution_providers)

    core.launch()

if __name__ == '__main__':
    main()

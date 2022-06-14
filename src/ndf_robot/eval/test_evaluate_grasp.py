from ndf_robot.eval.evaluate_grasp import EvaluateGraspParser

if __name__ == '__main__':
    parser = EvaluateGraspParser()
    parser.load_config('debug_config.yml')
    parser.create_model()

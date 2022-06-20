from ndf_robot.eval.evaluate_grasp import EvaluateGraspSetup, QueryPoints

if __name__ == '__main__':
    # setup = EvaluateGraspSetup()
    # setup.load_config('debug_config.yml')
    # setup.create_model()
    # setup.create_eval_dir('DEBUG')
    # print(setup.get_demo_load_dir())
    # print(setup.get_shapenet_obj_dir())
    rect = QueryPoints.generate_rect(500, 2, 2, 0.5, 4)
    print(rect)

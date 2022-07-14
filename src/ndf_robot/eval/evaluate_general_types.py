from ast import Name
from enum import Enum
from typing import NamedTuple

ModelTypes = {
    'CONV_OCC',
    'VNN_NDF',
}

QueryPointTypes = {
    'SPHERE',
    'RECT',
    'ARM',
    'SHELF',
}

# GripperPointTypes = {
#     'SPHERE',
#     'RECT',
# }

# RackQueryPointTypes = {
#     'ARM',
# }


class TrialResults(Enum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    BAD_OPT_POS = 2
    NO_FEASIBLE_IK = 3
    INTERSECTING_EE = 4
    GET_FEASIBLE_IK_FAILED = 5
    GET_IK_FAILED = 6
    COMPUTE_IK_FAILED = 7
    POST_PROCESS_FAILED = 8
    GET_PCD_FAILED = 9
    JOINT_PLAN_FAILED = 10
    GRASP_SUCCESS = 11
    DEBUG_FAILURE = 12
    PLACE_JOINT_PLAN_FAILED = 13


class RobotIDs:
    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10


class SimConstants:
    # General configs
    N_CAMERAS = 4

    PREGRASP_OFFSET_TF = [0, 0, 0.25, 0, 0, 0, 1]

    # PREPLACE_HORIZONTAL_OFFSET_TF = [0, -0.2, 0, 0, 0, 0, 1]
    PREPLACE_OFFSET_CLOSE_TF= [0, -0.042, 0.06, 0, 0, 0, 1]
    PREPLACE_OFFSET_FAR_TF = [0, -0.084, 0.12, 0, 0, 0, 1]
    # PREPLACE_OFFSET_TF = [0, -0.084, 0.12, 0, 0, 0, 1]

    # placement of table
    TABLE_POS = [0.5, 0.0, 0.4]
    TABLE_SCALING = 0.9
    # TABLE_Z = 1.05  # Was 1.15
    TABLE_Z = 1.15

    # Different from table so that more stuff is in focus
    CAMERA_FOCAL_Z = 1.15

    # placement of object
    # x is forward / back when facing robot
    # +y is right when facing robot

    OBJ_SAMPLE_X_LOW_HIGH = [0.4, 0.5]
    OBJ_SAMPLE_Y_LOW_HIGH = [-0.2, 0.2]
    # OBJ_SAMPLE_Y_LOW_HIGH = [-0.3, 0]
    OBJ_SAMPLE_Z_OFFSET = 0.0  # was 0.1
    OBJ_SAMPLE_R = 0.2  # was 0.2

    # Object scales
    MESH_SCALE_DEFAULT = 0.5
    MESH_SCALE_HIGH = 0.6
    MESH_SCALE_LOW = 0.4

    # MESH_SCALE_DEFAULT = 0.3
    # MESH_SCALE_HIGH = 0.35
    # MESH_SCALE_LOW = 0.175

    # Avoid Mugs
    MUG_AVOID_SHAPENET_IDS = {
        '32e197b8118b3ff6a4bd4f46ba404890',
        '7374ea7fee07f94c86032c4825f3450',
        '9196f53a0d4be2806ffeedd41ba624d6',
        'b9004dcda66abf95b99d2a3bbaea842a',
        '9ff8400080c77feac2ad6fd1941624c3',
        '4f9f31db3c3873692a6f53dd95fd4468',
        '1c3fccb84f1eeb97a3d0a41d6c77ec7c',
        'cc5b14ef71e87e9165ba97214ebde03',
        '159e56c18906830278d8f8c02c47cde0',
        'c6b24bf0a011b100d536be1f5e11c560',
        '9880097f723c98a9bd8c6965c4665b41',
        'e71102b6da1d63f3a363b55cbd344baa',
        '27119d9b2167080ec190cb14324769d',
        '89bd0dff1b386ebed6b30d74fff98ffd',
        '127944b6dabee1c9e20e92c5b8147e4a',
        '513c3410e8354e5211c7f3807925346a',
        '1bc5d303ff4d6e7e1113901b72a68e7c',
        'b98fa11a567f644344b25d683fe71de',
        'a3cd44bbd3ba5b019a4cbf5d3b79df06',
        'b815d7e084a5a75b8d543d7713b96a41',
        '645b0e2ef3b95979204df312eabf367f',
        '599e604a8265cc0a98765d8aa3638e70',
        '2997f21fa426e18a6ab1a25d0e8f3590',
        'c34718bd10e378186c6c61abcbd83e5a',
        'b7841572364fd9ce1249ffc39a0c3c0b',
        '604fcae9d93201d9d7f470ee20dce9e0',
        'e16a895052da87277f58c33b328479f4',
        '659192a6ba300f1f4293529704725d98',
        '3093367916fb5216823323ed0e090a6f',
        'c7f8d39c406fee941050b055aafa6fb8',
        '64a9d9f6774973ebc598d38a6a69ad2',
        '24b17537bce40695b3207096ecd79542',
        'a1d293f5cc20d01ad7f470ee20dce9e0',
        '6661c0b9b9b8450c4ee002d643e7b29e',
        '85d5e7be548357baee0fa456543b8166',
        'c2eacc521dd65bf7a1c742bb4ffef210',
        'bf2b5e941b43d030138af902bc222a59',
        '127944b6dabee1c9e20e92c5b8147e4a',
        'c2e411ed6061a25ef06800d5696e457f',
        '275729fcdc9bf1488afafc80c93c27a9',
        '642eb7c42ebedabd223d193f5a188983',
        '3a7439cfaa9af51faf1af397e14a566d',
        '642eb7c42ebedabd223d193f5a188983',
        '1038e4eac0e18dcce02ae6d2a21d494a',
        '7223820f07fd6b55e453535335057818',
        '141f1db25095b16dcfb3760e4293e310',
        '4815b8a6406494662a96924bce6ef687',
        '24651c3767aa5089e19f4cee87249aca',
        '5ef0c4f8c0884a24762241154bf230ce',
        '5310945bb21d74a41fabf3cbd0fc77bc',
        '6e884701bfddd1f71e1138649f4c219',
        '345d3e7252156db8d44ee24d6b5498e1',
        'a3cd44bbd3ba5b019a4cbf5d3b79df06',
        '24651c3767aa5089e19f4cee87249aca',
        'b7841572364fd9ce1249ffc39a0c3c0b',
        '1be6b2c84cdab826c043c2d07bb83fc8',
        '604fcae9d93201d9d7f470ee20dce9e0',
        '35ce7ede92198be2b759f7fb0032e59',
        'e71102b6da1d63f3a363b55cbd344baa',
        'dfa8a3a0c8a552b62bc8a44b22fcb3b9',
        'dfa8a3a0c8a552b62bc8a44b22fcb3b9',
        '4f9f31db3c3873692a6f53dd95fd4468',
        '10c2b3eac377b9084b3c42e318f3affc',
        '162201dfe14b73f0281365259d1cf342',
        '1a1c0a8d4bad82169f0594e65f756cf5',
        '3a7439cfaa9af51faf1af397e14a566d',
        '1f035aa5fc6da0983ecac81e09b15ea9',
        '83b41d719ea5af3f4dcd1df0d0a62a93',
        '3d3e993f7baa4d7ef1ff24a8b1564a36',
        '3c0467f96e26b8c6a93445a1757adf6',
        '414772162ef70ec29109ad7f9c200d62',
        '3093367916fb5216823323ed0e090a6f',
        '68f4428c0b38ae0e2469963e6d044dfe',
        'd0a3fdd33c7e1eb040bc4e38b9ba163e',
        'c7ddd93b15e30faae180a52fd2be32',
        '3c0467f96e26b8c6a93445a1757adf6',
        '89bd0dff1b386ebed6b30d74fff98ffd',
        '1dd8290a154f4b1534a8988fdcee4fde',
        '1ae1ba5dfb2a085247df6165146d5bbd',
        '9426e7aa67c83a4c3b51ab46b2f98f30',
        '35ce7ede92198be2b759f7fb0032e59',
        'bcb6be8f0ff4a51872e526c4f21dfca4',
        '43f94ba24d2f075c4d32a65fb7bf4ebc',
        'b9004dcda66abf95b99d2a3bbaea842a',
        '159e56c18906830278d8f8c02c47cde0',
        '275729fcdc9bf1488afafc80c93c27a9',
        '9196f53a0d4be2806ffeedd41ba624d6',
        '64a9d9f6774973ebc598d38a6a69ad2',
        '9880097f723c98a9bd8c6965c4665b41',
        '1dd8290a154f4b1534a8988fdcee4fde',
        '2037531c43448c3016329cbc378d2a2',
        '43f94ba24d2f075c4d32a65fb7bf4ebc',
        'b9f9f5b48ab1153626829c11d9aba173',
        '5582a89be131867846ebf4f1147c3f0f',
        '71ca4fc9c8c29fa8d5abaf84513415a2',
        'd32cd77c6630b77de47c0353c18d58e',
        '1ea9ea99ac8ed233bf355ac8109b9988',
        'c6b24bf0a011b100d536be1f5e11c560',
        'b98fa11a567f644344b25d683fe71de',
        'c82b9f1b98f044fc15cf6e5ad80f2da',
        '5b0c679eb8a2156c4314179664d18101',
        '546648204a20b712dfb0e477a80dcc95',
        'd309d5f8038df4121198791a5b8655c',
        '6c04c2eac973936523c841f9d5051936',
        '71ca4fc9c8c29fa8d5abaf84513415a2',
        '46955fddcc83a50f79b586547e543694',
        '659192a6ba300f1f4293529704725d98',
        'b9be7cfe653740eb7633a2dd89cec754',
        '9fc96d41ec7a66a6a159545213d74ea',
        '5582a89be131867846ebf4f1147c3f0f',
        'c2e411ed6061a25ef06800d5696e457f',
        '8aed972ea2b4a9019c3814eae0b8d399',
        'e363fa24c824d20ca363b55cbd344baa',
        '9426e7aa67c83a4c3b51ab46b2f98f30',
        '6661c0b9b9b8450c4ee002d643e7b29e',
        '8aed972ea2b4a9019c3814eae0b8d399',
        'c39fb75015184c2a0c7f097b1a1f7a5',
        '24b17537bce40695b3207096ecd79542',
        '83b41d719ea5af3f4dcd1df0d0a62a93',
        'c7ddd93b15e30faae180a52fd2be32',
        '46955fddcc83a50f79b586547e543694',
        'c82b9f1b98f044fc15cf6e5ad80f2da',
        'd32cd77c6630b77de47c0353c18d58e',
        '2037531c43448c3016329cbc378d2a2',
        '6500ccc65e210b14d829190312080ea3',
        '6c5ec193434326fd6fa82390eb12174f',
        '1bc5d303ff4d6e7e1113901b72a68e7c',
        '6d2657c640e97c4dd4c0c1a5a5d9a6b8',
        '6c5ec193434326fd6fa82390eb12174f',
        'f3a7f8198cc50c225f5e789acd4d1122',
        'f23a544c04e2f5ccb50d0c6a0c254040',
        'f42a9784d165ad2f5e723252788c3d6e',
        'ea33ad442b032208d778b73d04298f62',
        'ef24c302911bcde6ea6ff2182dd34668',
        'f99e19b8c4a729353deb88581ea8417a',
        'fd1f9e8add1fcbd123c841f9d5051936',
        'f626192a5930d6c712f0124e8fa3930b',
        'ea127b5b9ba0696967699ff4ba91a25',
        'f1866a48c2fc17f85b2ecd212557fda0',
        'ea95f7b57ff1573b9469314c979caef4',
        'b88bcf33f25c6cb15b4f129f868dedb'
    }


class OLDTrialData():
    """
    Named container class for trial specific information

    Args:
        grasp_success (bool): True if trial was successful
        trial_result (TrialResults): What the outcome of the trial was
            (including) failure modes
        obj_shapenet_id (str): Shapenet id of object used in trial
    """
    grasp_success = False
    place_success_teleport = False
    trial_result = TrialResults.UNKNOWN_FAILURE
    obj_shapenet_id = None
    best_grasp_idx = -1
    best_place_idx = -1


class TrialData():
    obj_shapenet_id: str
    trial_result: TrialResults
    best_opt_idx: int


class ExperimentTypes(Enum):
    GRASP = 0
    RACK_PLACE_TELEPORT = 1
    SHELF_PLACE_TELEPORT = 2
    RACK_PLACE_GRASP = 3

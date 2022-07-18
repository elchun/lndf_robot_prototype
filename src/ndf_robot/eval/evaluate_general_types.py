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
    # PLACE_JOINT_PLAN_FAILED = 13


class RobotIDs:
    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10


class SimConstants:
    # General configs
    N_CAMERAS = 4

    PREGRASP_OFFSET_TF = [0, 0, 0.25, 0, 0, 0, 1]

    # PREPLACE_HORIZONTAL_OFFSET_TF = [0, -0.2, 0, 0, 0, 0, 1]
    # PREPLACE_HORIZONTAL_OFFSET_TF = [0.012, -0.242, 0.06, 0, 0, 0, 1]
    PREPLACE_HORIZONTAL_OFFSET_TF = [0.012, -0.092, 0.06, 0, 0, 0, 1]
    PREPLACE_OFFSET_CLOSE_TF = [0.012, -0.042, 0.06, 0, 0, 0, 1]
    PREPLACE_OFFSET_FAR_TF = [0.012, -0.084, 0.12, 0, 0, 0, 1] # Not currently used
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

    BOWL_AVOID_SHAPENET_IDS = {
        'e072da5c1e38c11a7548281e465c9303',
        '571f294b5ab7e47a5517442d05421cb',
        '7d7bdea515818eb844638317e9e4ff18',
        'eff9864bfb9920b521374fbf1ea544c',
        'ea473a79fd2c98e5789eafad9d8a9394',
        'e4c871d1d5e3c49844b2fa2cac0778f5',
        'f6ffca90c543edf9d6438d5cb8c578c6',
        'f6ffca90c543edf9d6438d5cb8c578c6'
    }

    # # Test mugs (withheld from training data)
    # MUG_TEST_SHAPENET_IDS = {
    #     'd46b98f63a017578ea456f4bbbc96af9',
    #     'd75af64aa166c24eacbe2257d0988c9c',
    #     'd7ba704184d424dfd56d9106430c3fe',
    #     'daee5cf285b8d210eeb8d422649e5f2b',
    #     'dcec634f18e12427c2c72e575af174cd',
    #     'df026976dc03197213ac44947b92716e',
    #     'e6dedae946ff5265a95fb60c110b25aa',
    #     'e9499e4a9f632725d6e865157050a80e',
    #     'e94e46bc5833f2f5e57b873e4f3ef3a4',
    #     'e984fd7e97c2be347eaeab1f0c9120b7',
    #     'e9bd4ee553eb35c1d5ccc40b510e4bd',
    #     'ec846432f3ebedf0a6f32a8797e3b9e9',
    #     'edaf960fb6afdadc4cebc4b5998de5d0',
    #     'eecb13f61a93b4048f58d8b19de93f99',
    #     'f09e51579600cfbb88b651d2e4ea0846',
    #     'f1c5b9bb744afd96d6e1954365b10b52',
    #     'f1e439307b834015770a0ff1161fa15a',
    #     'f7d776fd68b126f23b67070c4a034f08',
    #     'fad118b32085f3f2c2c72e575af174cd',
    #     'ff1a44e1c1785d618bca309f2c51966a'
    # }

    # BOWL_TEST_SHAPENET_IDS = {
    #     'f2ef5e5b49f2bb8340dfb1e6c8f5a333',
    #     '34875f8448f98813a2c59a4d90e63212',
    #     'f0fdca5f5c7a06252dbdfbe028032489',
    #     'a0b34a720dd364d9ccdca257be791a55',
    #     '2d2c419136447fe667964ba700cd97f5',
    #     'f09d5d7ed64b6585eb6db0b349a2b804',
    #     'bd2ba805bf1739cdedd852e9640b8d4',
    #     '64d7f5eb886cfa48ce6101c7990e61d4',
    #     'ea473a79fd2c98e5789eafad9d8a9394',
    #     'ee3b4a98683feab4633d74df68189e22',
    #     'e30e5cbc54a62b023c143af07c12991a',
    #     'a1d26a16a0caa78243f1c519d66bb167',
    #     'faa200741fa93abb47ec7417da5d353d',
    #     'fa23aa60ec51c8e4c40fe5637f0a27e1',
    #     '2e545ccae1cdc69d879b85bd5ada6e71',
    #     '2c1df84ec01cea4e525b133235812833',
    #     'e3e57a94be495771f54e1b6f41fdd78a',
    #     'e816066ac8281e2ecf70f9641eb97702',
    #     'f2cb15fb793e7fa244057c222118625',
    #     '188281000adddc9977981b941eb4f5d1',
    #     '7995c6a5838e12ed447eea2e92abe28f',
    #     'eff9864bfb9920b521374fbf1ea544c',
    #     'fa61e604661d4aa66658ecd96794a1cd',
    #     'f44387d8cb8d2e4ebaedc225f2279ecf',
    #     'ed220bdfa852f00ba2c59a4d90e63212',
    #     'ff7c33db3598df342d88c45db31bc366',
    #     '32f9c710e264388e2150a45ec52bcbd7',
    #     'e3d4d57aea714a88669ff09d7001bab6',
    #     'e4c871d1d5e3c49844b2fa2cac0778f5',
    #     'fc77ad0828db2caa533e44d90297dd6e'
    # }

    # Train mugs:
    MUG_TRAIN_SHAPENET_IDS = {
        '10f6e09036350e92b3f21f1137c3c347',
        '128ecbc10df5b05d96eaf1340564a4de',
        '1305b9266d38eb4d9f818dd0aa1a251',
        '15bd6225c209a8e3654b0ce7754570c8',
        '17952a204c0a9f526c69dceb67157a66',
        '187859d3c3a2fd23f54e1b6f41fdd78a',
        '1a97f3c83016abca21d0de04f408950f',
        '1c9f9e25c654cbca3c71bf3f4dd78475',
        '1d18255a04d22794e521eeb8bb14c5b3',
        '1eaf8db2dd2b710c7d5b1b70ae595e60',
        '214dbcace712e49de195a69ef7c885a4',
        '2852b888abae54b0e3523e99fd841f4',
        '28f1e7bc572a633cb9946438ed40eeb9',
        '2d10421716b16580e45ef4135c266a12',
        '3143a4accdc23349cac584186c95ce9b',
        '336122c3105440d193e42e2720468bf0',
        '34869e23f9fdee027528ae0782b54aae',
        '34ae0b61b0d8aaf2d7b20fded0142d7a',
        '37f56901a07da69dac6b8e58caf61f95',
        '387b695db51190d3be276203d0b1a33f',
        '39361b14ba19303ee42cfae782879837',
        '3d1754b7cb46c0ce5c8081810641ef6',
        '403fb4eb4fc6235adf0c7dbe7f8f4c8e',
        '43e1cabc5dd2fa91fffc97a61124b1a9',
        '44f9c4e1ea3532b8d7b20fded0142d7a',
        '46ed9dad0440c043d33646b0990bb4a',
        '48e260a614c0fd4434a8988fdcee4fde',
        '4b7888feea81219ab5f4a9188bfa0ef6',
        '4b8b10d03552e0891898dfa8eb8eefff',
        '4d9764afa3fbeb1b6c69dceb67157a66',
        '52273f4b17505e898ef19a48ac4fcfdf',
        '54f2d6a0b431839c99785666a0fe0255',
        '57f73714cbc425e44ae022a8f6e258a7',
        '586e67c53f181dc22adf8abaa25e0215',
        '5c48d471200d2bf16e8a121e6886e18d',
        '5c7c4cb503a757147dbda56eabff0c47',
        '5d72df6bc7e93e6dd0cd466c08863ebd',
        '5fe74baba21bba7ca4eec1b19b3a18f8',
        '61c10dccfa8e508e2d66cbf6a91063',
        '62684ad0321b35189a3501eead248b52',
        '633379db14d4d2b287dd60af81c93a3c',
        '639a1f7d09d23ea37d70172a29ade99a',
        '649a51c711dc7f3b32e150233fdd42e9',
        '67b9abb424cf22a22d7082a28b056a5',
        '6aec84952a5ffcf33f60d03e1cb068dc',
        '6c379385bf0a23ffdec712af445786fe',
        '6dd59cc1130a426571215a0b56898e5e',
        '6faf1f04bde838e477f883dde7397db2',
        '71995893d717598c9de7b195ccfa970',
        '73b8b6456221f4ea20d3c05c08e26f',
        '79e673336e836d1333becb3a9550cbb1',
        '7a8ea24474846c5c2f23d8349a133d2b',
        '7d282cc3cedd40c8b5c4f4801d3aada',
        '7d6baadd51d0703455da767dfc5b748e',
        '8012f52dd0a4d2f718a93a45bf780820',
        '83827973c79ca7631c9ec1e03e401f54',
        '8570d9a8d24cb0acbebd3c0c0c70fb03',
        '85a2511c375b5b32f72755048bac3f96',
        '896f1d494bac0ebcdec712af445786fe',
        '8b1dca1414ba88cb91986c63a4d7a99a',
        '8b780e521c906eaf95a4f7ae0be930ac',
        '8f550770386a6cf82f23d8349a133d2b',
        '8f6c86feaa74698d5c91ee20ade72edc',
        '91f90c3a50410c0dc27effd6fd6b7eb0',
        '9278005254c8db7e95f577622f465c85',
        '928a383f79698c3fb6d9bc28c8d8a2c4',
        '92d6394732e6058d4bcbafcc905a9b98',
        '962883677a586bd84a60c1a189046dd1',
        '9737c77d3263062b8ca7a0a01bcd55b6',
        '9961ccbafd59cb03fe36eb2ab5fa00e0',
        '99eaa69cf6fe8811dec712af445786fe',
        '9af98540f45411467246665d3d3724c',
        '9c930a8a3411f069e7f67f334aa9295c',
        '9d8c711750a73b06ad1d789f3b2120d0',
        'a0c78f254b037f88933dc172307a6bb9',
        'a637500654ca8d16c97cfc3e8a6b1d16',
        'a6d9f9ae39728831808951ff5fb582ac',
        'a8f7a0edd3edc3299e54b4084dc33544',
        'b18bf84bcd457ffbc2868ebdda32b945',
        'b46e89995f4f9cc5161e440f04bd2a2',
        'b4ae56d6638d5338de671f28c83d2dcb',
        'b6f30c63c946c286cf6897d8875cfd5e',
        'b7e705de46ebdcc14af54ba5738cb1c5',
        'b811555ccf5ef6c4948fa2daa427fe1f',
        'b88bcf33f25c6cb15b4f129f868dedb',
        'bea77759a3e5f9037ae0031c221d81a4',
        'bed29baf625ce9145b68309557f3a78c',
        'c0c130c04edabc657c2b66248f91b3d8',
        'c51b79493419eccdc1584fff35347dc6',
        'c60f62684618cb52a4136492f17b9a59',
        'c6bc2c9770a59b5ddd195661813efe58',
        'ca198dc3f7dc0cacec6338171298c66b',
        'cf777e14ca2c7a19b4aad3cc5ce7ee8',
        'd38295b8d83e8cdec712af445786fe'
    }

    BOWL_TRAIN_SHAPENET_IDS = {
        '13e879cb517784a63a4b07a265cff347',
        '77301246b265a4d3a538bf55f6b58cef',
        'afb6bf20c56e86f3d8fdbcba78c84028',
        'c82e28d7f713f07a5a15f0bff2482ab8',
        '9a6cec1cfca7e8b5ebc583f22bd58b85',
        '441cac4bae5c7c315c2a2c970803cfe2',
        'a9ba34614bfd8ca9938afc5c0b5b182',
        '879a5c740b25ef5c7a88a2ad67bfd073',
        '5e2c558555092b706e30831c34845769',
        '5563324c9902f243a2c59a4d90e63212',
        '571f294b5ab7e47a5517442d05421cb',
        'cda15ee9ad73f9d4661dc36b3dd991aa',
        '4530e6df2747b643f6415fd62314b5ed',
        'cfac22c8ca3339b83ce5cb00b21d9584',
        '538db91266f2829bc0f7e0e05bae6131',
        'aeec00d8dd6e196451a2ad3de1977657',
        '9dcfab1fab6232003bec56bff764ba78',
        '454fa7fd637177cf2bea4b6e7618432',
        '2ffe06ee50ec1420adbe0813683fcfd0',
        '45603bffc6a2866b5d1ac0d5489f7d84',
        'b69e25e7ab9b829215b14098c484b7c1',
        'b5d81a5bbbb8efe7c785f06f424b9d06',
        'da5f13f4048dbd72fcb8d8c6d4df8143',
        '98d3408054ab409fd261c3d31246fed3',
        'a73d531b90965199e5f6d587dbc348b5',
        'c6be3b333b1f7ec9d42a2a5a47e9ed5',
        '38d2122b2aa05f92e1b9b7b3765e2bea',
        '12ddb18397a816c8948bef6886fb4ac',
        'ae5c7d8a453d3ef736b0f2a1430e993a',
        '899af991203577f019790c8746d79a6f',
        '36ca3b684dbb9c159599371049c32d38',
        '92f04b8d36c98d18221d647283ba1e26',
        'aa70f5e55fa61d8dac6b8e58caf61f95',
        '5bb12905529c85359d3d767e1bc88d65',
        'e16e27d3670853b12d4e6b2984840098',
        '530c2d8f55f9e9e8ce364511e87f52b0',
        'aeb7b4bb33fd43a14e23e9314af9ae57',
        '8b90aa9f4418c75452dd9cc5bac31c96',
        'c0f57c7f98d2581c744a455c7eef0ae5',
        '6816c12cc00a961758463a756b0921b5',
        'ab2fd38fc4f37cce86bbb74f0f607cdd',
        '2a1e9b5c0cead676b8183a4a81361b94',
        '9ea66797afeb86196ea1c514a0de6d2d',
        '468b9b34d07eb9211c75d484f9069623',
        '95ac294f47fd7d87e0b49f27ced29e3',
        'd162f269bc5fb7da85df81f999944b5d',
        '3a7737c7bb4a6194f60bf3def77dca62',
        'a83b2eb519fbef17e922c0c19528ec07',
        '381db800b87b5ff88616812464c86290',
        '8d457deaf22394da65c5c31ac688ec4',
        'd78860e8318fb75a12a72bbc636a1f9d',
        '3bedfd655c45c6747bc26b16f550876f',
        'a08af31cb43f1f662d88c45db31bc366',
        '80608e58db79d4d83b722c86abee0751',
        'b941ca9770b59c3918a27ff49f2f297f',
        '524eb99c8d45eb01664b3b9b23ddfcbc',
        '64b180d51c6b8cc1e4e346ee2650d150',
        'b65cbede8f483d51f7de4d28691515e1',
        '260545503087dc5186810055962d0a91',
        'bec41cc7f631e00b1bf40e92cbe8569f',
        '1f910faf81555f8e664b3b9b23ddfcbc',
        'b195dbe537993945e4e346ee2650d150',
        '4b32d2c623b54dd4fe296ad57d60d898',
        'a95e0d8b37f8ca436a3309b77df3f951',
        '154ab09c67b9d04fb4971a63df4b1d36',
        'be3c2533130dd3da55f46d55537192b6',
        '4eefe941048189bdb8046e84ebdc62d2',
        'a2841256fc6fd90da518b93bb7233de6',
        '18529eba21e4be8b5cc4957a8e7226be',
        '4cf18594e8851a3d3a5e6305a3a7adee',
        '709778e5ebb67e32b396efafc0d0d8ac',
        'a593e8863200fdb0664b3b9b23ddfcbc',
        'd2e1dc9ee02834c71621c7edb823fc53',
        'd28f7a7a8fbc5fc925b5a13384fa548b',
        '8d75c3c065fa3c7055f46d55537192b6',
        '782aa72e7f14975b39d764edb37837d3',
        'c2882316451828fd7945873d861da519',
        '3f56833e91054d2518e800f0d88f9019',
        'c25fd49b75c12ef86bbb74f0f607cdd',
        '2cfecc8ce6c7cd1f3497637ec59e0374',
        '5f2ef71aa9b94edbb84959963148f2a2',
        'ad8a50aa5894bc40c762eb31f1f0c676',
        'c1bad5cc2d9050e48aee29ee398d36ed',
        '47175c7b12bf0b61320b691da6d871fb',
        'bbf4b10b538c7d03bcbbc78f3e874841',
        'a5f42bbfddcadf05eeb8d422649e5f2b',
        '754634abeb7bffc32977b68653eb2e1e',
        '8bb057d18e2fcc4779368d1198f406e7',
        'dd381b3459767f7b18f18cdcd25d1bbb',
        '519f07b4ecb0b82ed82a7d8f544ae151',
        '708fce7ba7d911f3d5b5e7c77f0efc2',
        '7d7bdea515818eb844638317e9e4ff18',
        'dd15ee7374954827c5be64abd4416cc9',
        '446583d7bf857dced5cb6d178687b980',
        '6118da3aa071921b664b3b9b23ddfcbc',
        '817221e45b63cef62f74bdafe5239fba',
        '292d2dda9923752f3e275dc4ab785b9f',
        '24907349b2323824664b3b9b23ddfcbc',
        'b4c43b75d951401631f299e87625dbae',
        'd5b3fb99c8084a6c5430c0f7a585e679',
        '6a772d12b98ab61dc26651d9d35b77ca',
        'd98455f19960f99ed684faddec3c0090',
        '53f5240e1e82c96e2d20e9f11baa5f8f',
        '3f6a6718d729b77bed2eab6efdeec5f8',
        'ce48ffb418b99996912a38ce5826ebb8',
        '960c5c5bff2d3a4bbced73c51e99f8b2',
        'baeaa576ba746179e8d7df9e0b95a9b2',
        '6494761a8a0461777cba8364368aa1d',
        '9c7edf31042fea84df95dda4222cd1bf',
        'a1393437aac09108d627bfab5d10d45d',
        '4017528ab5816338664b3b9b23ddfcbc',
        '1a0a2715462499fbf9029695a3277412',
        '6930c4d2e7e880b2e20e92c5b8147e4a',
        '8d1f575e9223b28b8183a4a81361b94',
        '3152c7a0e8ee4356314eed4e88b74a21',
        'a98b76254bcd32c05481b689f582aa44',
        'c8a4719150446e84664b3b9b23ddfcbc',
        '429a622eac559887bbe43d356df0e955',
        '9a52843cc89cd208362be90aaa182ec6',
        '4fdb0bd89c490108b8c8761d8f1966ba',
        'ce905d4381d4daf65287b12a83c64b85',
        'e3095ecacc36080cb398a1cfd1079875',
        '11547e8d8f143557525b133235812833',
        '68582543c4c6d0bccfdfe3f21f42a111',
        'b5d1f064841b476dba5342d638d0c267',
        'aad3cd6a40a15cb8664b3b9b23ddfcbc',
        '4845731dbf7522b07492cbf7d8bec255',
        '89bde5319b8a7de044841730d607e227',
        '46d014209027ec188e74a3351a4d8b3a',
        'baa4d6ca9521a55b51d6f7c8e810987e',
        'c399bfee7f25f0ff95aab58c1db71c10',
        '4227b58665eadcefc0dc3ed657ab97f0',
        '5b6d840652f0050061d624c546a68fec',
        'd1addad5931dd337713f2e93cbeac35d',
        '5aad71b5e6cb3967674684c50f1db165',
        '7c43116dbe35797aea5000d9d3be7992',
        'db180e4f9a75ae717ba6f8f10959534c',
        '4967063fc3673caa47fe6b02985fbd0',
        '804092c488e7c8e420d3c05c08e26f',
        '4bf56e6a081c2a85a11f6bacf5c7662d',
        'bed9f618bef8d3e9f6d436d018e2b50f',
        'be8a2f8460a963426a6acc5f155f864',
        '6501113f772fc798db649551c356c6e8',
        'b007af6eabaa290dd42b9650f19dd425',
        'a042621b3378bc18a2c59a4d90e63212',
        'a29f53390194166665c638ab0bc6bc66',
        '9024177b7ed352f45126f17934e17803',
        '1fbb9f70d081630e638b4be15b07b442',
        'a0ac0c76dbb4b7685430c0f7a585e679',
        'e072da5c1e38c11a7548281e465c9303',
        '1b4d7803a3298f8477bdcb8816a3fac9',
        'bb7586ebee0dc2be4e346ee2650d150',
        '7ed1eb89d5463580a2c59a4d90e63212',
        '5019f979a6a360963a5e6305a3a7adee',
        '56803af65db53307467ca2ad6571afff',
        '326c6280eee2bf74257e043ec9630f'
    }




class OLDTrialData():
    """
    Named container class for trial specific information
    NOT CURRENTLY USED

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
    obj_shapenet_id: str  # Shapenet id of object used.
    trial_result: TrialResults  # Result of each trial.
    # best_opt_idx: int
    aux_data: dict  # For experiment specific data.


class ExperimentTypes(Enum):
    GRASP = 0
    RACK_PLACE_TELEPORT = 1
    SHELF_PLACE_TELEPORT = 2
    RACK_PLACE_GRASP = 3

False
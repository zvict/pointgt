#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import signal
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_REPO = "victor678/pointgt"
#: Model repo, matching the convention used for this group's other releases.
DEFAULT_REPO_TYPE = "model"
DEFAULT_REVISION = "main"

RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

_CHUNK = 1 << 20


@dataclass(frozen=True)
class Artifact:
    """One downloadable file.

    ``path`` is simultaneously the path inside the HF repo and the path under ``--dest``.
    Keeping them identical is what lets ``huggingface_hub``'s ``local_dir`` mode and the plain
    HTTPS fallback write to the same place with no per-file mapping.
    """

    path: str
    sha256: str
    size: int
    group: str
    scene: str
    note: str = ""

    @property
    def pending(self) -> bool:
        """True when the release has no measured hash for this file yet."""
        return not self.sha256


def _dress_frames() -> tuple[Artifact, ...]:
    """The 35-frame RBF-deformed control sequence consumed by ``render_edit.py --pcd_dir``.

    These are the *fitted* clouds from the training run (10383 points, one per point of the
    checkpoint's cloud), not the 15000-point Blender export in the dataset.  ``render_edit.py``
    checks the count and refuses a mismatch, so the distinction matters.
    """
    hashes = (
        "a15c7aa8370308bd49b242259e6191c15fb958861a2f3df65a3cd333b60df51f",
        "01149ab3a17df04d99207511f7dc5fceaa6e81c36b9fc9412134f4ee27d45f41",
        "0c2f2f3d89245c7b8a6dd261e27bcf59de725b8e3e3ff17d9f3f851804e5a6cf",
        "3d55d65e2b61777430159bd6439902e3cc8c1828addaec323c4400a4918136f2",
        "7f8dfa9d1e58e47eb84abc077903a8326df39a98325a1a49e8953d72e09496dc",
        "05a2fe6e976ea8aed4e9cab09d89d5669748a93772519ede4ec772cb77861685",
        "fa2aab867b7f0042d6f621be68b8713d353b1122356d0c5d760c32b6d75d0cec",
        "c4051ba6257220f72f793869056377cf371e8d1936cd50b2d455fe5f13c0bcd9",
        "208106ff980b4cd953808b0396bf7be9a9d30a1feba491419b0549fbda111025",
        "88aca14cc082ad33912a5cf263ea3766c0779d99d0f0086439897dbc53b624d9",
        "08ee06b075d9a73ed7edd567f36224db1b124abe1dc9f6d0559563e5ad025e14",
        "be469af3d3c58171b3e7ef8328f979ea98a7dd06b4aa2aaa3295b2008cf5ef1d",
        "209aa72c44d223f509a8862ce3c6ccf86e15fe97e330f1f3fae3f374eaaa1f35",
        "7234b818207cac697807499ebc3443c181edeed1970bd12d090137c80afb2035",
        "b3655268862240d8bfcdcaf6985a503e39e6b55082e6d16340f290c27e7ab821",
        "2e4d27872e1270fe7487d5b5df7ae389b057950a131d92ca47d5cb7493ed2e64",
        "21021757513d16e9ef4e64717851b4d02dd35d606b49616f40a75f4ba602b709",
        "2c5de41c3b87d1efdc148352ed5c764b0da6bef78ab06c756f3134da08e4dddf",
        "3249b00453da2a7d0b36f0ef2c7f03ef4bd65c8525cc2e93287222b5c54fd801",
        "738ddcec03440bfd571267bac9f9f8f53f504ba56fac8a9bc730a9a5fe74f6cb",
        "6eb3148d699be200cc4cdb3d51af93f31ad350fdfc796efca9c9030dde4d3d22",
        "b4f309f9b7037b261b2d5148322214b105dc76852d5bc1624b0da755e30aa1a3",
        "5e7f690c6e0ec184e4466d8933916f0ad00b0e9063231fc1b574837c03005b18",
        "b7445140777b55eb93c9290920291651dfef00837f78977fb98e1b71df16e891",
        "beb218471338fff81051cf7eef06383f8944d3a28a2839fdfc9bb0890cb86bb9",
        "03d985a9417a03f5c37080d64d16f3b8da481d8ed9776cfc791e88b23a9f0308",
        "e596d91c764a84dc90354bd45015fd4599b65a8edb0194f8fc46290f293dc00b",
        "d68316a440408043e634972900b086580942702a1492352498225a8eb6fd432a",
        "5664a854de3a25a46e138c3797aaab019bb2997997b055c80f2b6e61c9c264b1",
        "6340c5153ea56051d7d284273354fb09f0149050595de321d27a594478f6478c",
        "a04e43e4d4ac6360776183232371c3e31b05f47940aa64e303bb1dcef0d3abc7",
        "b70ee9f20b9cafeccc483b56150f8abf487e7ca9469660de55dbf68021226d5d",
        "c201f732138b8c1264cff960a287a988815a566b8e8b364e2d2e76adea6a7434",
        "1fa76c25e963b0f964c06f1f475a961f91ed70bf918f4c39fdd228e45d126c25",
        "c42b478f09e4b9f9ff68420c60b64e1ae19ddd540303f2dbfb0b434aed83f5d7",
    )
    return tuple(
        Artifact(
            path=f"demo/dress/rbf_pcds/edit_{i:04d}.ply",
            sha256=digest,
            size=249340,
            group="demo",
            scene="dress",
            note="render_edit.py --pcd_dir demo/dress/rbf_pcds",
        )
        for i, digest in enumerate(hashes, start=1)
    )


def _dress_dataset() -> tuple[Artifact, ...]:
    """The renders and camera poses ``configs/editing/dress_*.yml`` train and test on.

    ``dataset.path``, ``eval.dataset.path`` and ``test.datasets[0].path`` all point at
    ``data/sketchfab/dress1``, and :mod:`dataset.load_nerfsyn` opens exactly two things there:
    ``transforms_{split}.json`` and the ``r_<i>.png`` each frame's ``file_path`` names. Those are
    what is listed here; the rest of the archived scene directory (the source meshes, the
    ``dress-edit*`` sequences, a second copy of the renders under ``renderings/``) is not read by
    anything in this release and is not shipped.

    Redistributable: every file is a render of, or was derived from, "Dress with gold leaves" by
    Canvastique3D, CC BY 4.0 -- see the NOTICE written beside the demo. Attribution is the whole
    obligation, and :data:`DRESS_NOTICE` carries it.

    The two transforms files are byte-identical in the archive: this scene's "test" split is its
    train split, so its metrics are reconstruction on seen views, not held-out novel views.
    """
    frames = (
    ("7e68ab9df08ee401048bde998bf1fa4622b1c914c26a9ef99686be421715165b", 285666),
    ("0b8921099f22a2d965b7b9b1d064f95202b7680032986a946f28c85ec68f1d58", 252875),
    ("51ac41bc64e09fde5e8070e42509cace4d53a119d2ed35a9f6addaee32374a18", 226526),
    ("8d225beae1920daf64810a3abbeb287234108b68a6df597681d02cc20f091591", 308311),
    ("d43cfc2dd80475e7096740c2873e74b048d30bef5087c657a898a64bec13418c", 226780),
    ("1b58d5d9eafba0686970c54996d9e2295454193c1ac271223f4aaba06f442393", 175592),
    ("65c0f2deddfe8b52039dae0cd4055d3f9338ae6b2d9e57f716fbfd5ed523a226", 194287),
    ("fd412b8dff4edfc4d77e4ce7f7197ce9ddee95ed7a3be8c63974deeb3d09caaa", 196607),
    ("21187cd9049e56a8394f5c5cb2373022d09f63fca42c8a3063f610da4bbaf070", 174703),
    ("90c33960f342a749e9a3167fb50acc67063d17253905a1d8f8cd30e55912fd35", 214604),
    ("0e3b570f1546c25def09cc740dcf5d46256eea0b932c776b7b384147ee39ec29", 314460),
    ("d92ba08492b7f77b190af4eb2ee22ef94a9096ff3af2764e2b9f2b1bedddc8bf", 277230),
    ("5fd1f85c359074275511e11f350eeb84f64dccf3b669c2c9badb69334e1c5f67", 228521),
    ("c2e19bb17330a38db68590d266ae75b4a8a2775b61a7119232be5c09826ce191", 322622),
    ("fcf8498ef82cf0293dc80516bd6b16489ae60d2ee858e177551af562d7385343", 247746),
    ("40dd46ccbaa5dc27f01bb85c28a92d63076495b9cca7c0645f4b85065e6ce554", 211933),
    ("44bef98058b126988a11a3d8cd5e02108b3a4dea458d0fd7f7052a3e99cf6be8", 303539),
    ("29f8a1422469f48ca27980b7a23dc50214e3de3410dd3fe0c685758f1bc289b1", 227944),
    ("b39adcbedc851d338a849cd869ca4fcf1ae5245606e30a69b8172c84acdd4b59", 253257),
    ("3c0a55e18788156fc959e3151fe1dfb32b32615789091e7cb388a4c7132d17d7", 220393),
    ("d95de462d000edb170f6b09100556958f94872279b27b9df9ff2a3a472c6a86b", 244284),
    ("6e4a81a0ccfae3be7b6d7bed1fbd13d600879b1e9fe0028e9a02eea57383bd15", 263613),
    ("1fae7dcda4d4b10ed01af0d04fb1f51378f068168c6880e081855fdf1a3fb74d", 186423),
    ("7ce113aa0b9e8e267b3a0747fb710128f14d89fdb2965468dc29fd29b2d72875", 245229),
    ("cf57dbd55c2960df6333a7eb7db4fdcf87613013bcf8f5d7ce1213de346cc8fd", 155610),
    ("3f4b8ab1c7d0fe89cbb26245fe38225b21648854078d970303de39d95b6571a6", 146706),
    ("2271f57b391c86c6f483a47e9c4ac3a57d49425b39a3fb4edddc24c8cbf7aee6", 268142),
    ("6a85642b2c5165cfa519db2c8f568e2a74c26f0f185d2c2670427c18e3626953", 230282),
    ("c5640df2d9f6ab46f8f5a4ff231f4afd90086d28c093a5816e9c17b1101c8baf", 307360),
    ("256a3d7f194386ae85cf5780d7bda1bb0e146777cea8a5698012ba43807f710c", 160908),
    ("e7feb34c3d255f4a2d779502dde5713b5790b814da264c961396f510339704b9", 209797),
    ("a494cdfc75e825c8938eac462ad279ca39cd3bce8a73768ac7c8a60f850ae546", 198977),
    ("09ec8aac3f34d4e3e01e31f72450a51fa3a436b9a04e9812201f5c32fe118bd2", 225083),
    ("a38524b6bf86332255cddf6e25d7ae8ba40aaa5b11360c02a43b28c5a669092f", 124977),
    ("5583e3f94daf320cc37ea566eb7119c099b801cfd9b273c4ca052d235affc14e", 223005),
    ("7bc4b700c053f79bb8cab3216051bbb48462ff91c6250fcde74442bfc9203166", 300763),
    ("b89b6de29528c1af0be1cc76134b1a5c42dc0288a72a39aefc63413edcbc654c", 306527),
    ("c1e10a6cbdf83c6d6cdc701e86dbdd5311f48e24db9d588ff1dadb4f01324153", 229387),
    ("32ae5930caf33646d0aebcfa7529ae69cba4fa5b184c6568930391c9238eda39", 194680),
    ("75883e2fa05700fa5a465fabf7f643789c1871257556a59809014c4a9b5c9352", 229753),
    ("f9434ed5b04c689b6a3fc2ea63d97bcdbe35d02536cce5460b3ac4287d0f1675", 209786),
    ("03b53eb9c73955c7697458df08dbd1488e28cb34946e0f5dc0f46dea12204ea7", 289341),
    ("6ac58ca0c057ff16ed7c9ca61e048b5232b693712a1c43baff415721ed60599d", 162120),
    ("ae49551c5474689b27caebadbb638faedd73d3476b8fa1598050ce4301508146", 318551),
    ("245c457f04f07ff5ffedaea43492aae41e6a6ff7bed968fbab60a56d14a71687", 272697),
    ("4549527992cdb9446b7a874a54f8ee80b9f26166585af58b0fd707057100966f", 251267),
    ("1a2bf4ed10a4eb483644e8bd3086abea351321b4e4428f73af36fc2ca0a5c7cc", 230481),
    ("18748c6d42b3baf86f8734fa7191e150601e40842ce40067905be20b0709a438", 231635),
    ("bc7bc2fae5717e6f6b2e655c2873401b405add1fa453e1578bf164b2fbd40e19", 188552),
    ("2114719228c4961ff3640d14ee969ba499aeb690901d81d4aa0d1fa2395495fe", 145430),
    ("cfae82f089c13e6346b7f3b1198ffa70d7401defa1c64ee4001103c5f6c2ec8f", 177193),
    ("92621684ebd88840c21804dae4c71de31f8a9aa2bb933ff421a4c7a8b590e44f", 224943),
    ("3ec0429af511a4a473fc74212a2b1957eaad8f1786b7bc41edcaf63daaa5c011", 295620),
    ("2aad8df74eaa5d986dae01f49963cc7e36b8c0cc1083f636467c88345ea6b2bc", 223894),
    ("a438114ed14a700b69d2a7a02960b9d6785d44022d0efad6100b8cb63ead7f6e", 207270),
    ("e183b01bed2449529a2347aa8f4976654cb759d5e73379c7215286b28644b61e", 261745),
    ("a7201999af1ffa9f989c5dfe17c86c7069d782f112de899ff9337acb5d29bd2f", 304450),
    ("ad439df51f25c5f472a7f203c37fad7eeabe615ba7753dbdf94262f35f5848d8", 137886),
    ("c399a6f2836e58b447a0559d8fac46df7eb8512f60745b56d21196bae1f39654", 271040),
    ("970ab6533af7ff738af6a6250787fef7bba0bc599616e8ee2d1315c723c20a59", 143307),
    ("46e56ee2439d4d089485159be4ea3eb161ab0e02fbc376fb878d6a9e74e973e8", 282804),
    ("827767a8e4faf95c4e0ec10ad6cdb31e130f74ec00fc1b07f898d8af6bdccbe0", 283616),
    ("915087404f1d8924d8783ee15672d97b74aa5afd67ff05cc279e26cbe02cb081", 203461),
    ("861d72e2c554fba026e50a503b81d9743aa34f02d68ed658cacef424f9238999", 205637),
    ("a9887b9bac7b8bca002730221242c4c37546aabe8dba975d82785d71aafe907e", 204567),
    ("6aa8158aced23427c844872c14bc92f0b9c10c4e09ff3ed9b4b2409072eb281c", 219278),
    ("3f4e693bbc1134f1b0ebdc165d87cc9acf3659d9f276ac66391cc4b9766273e0", 208699),
    ("94e8235ea845df74c63d5b7e92938581d736bea8dbcdc90a51afa7580919c35f", 200413),
    ("fe415605b093e42b7e5425ebcc9341640d540d0fce9977c89ed8502b886df405", 169093),
    ("9e38208215790d04c52b6b1a80e9802338b75725738f3b402ff177ad9bb05213", 237516),
    ("0fa0abec14d3bcb43340ef7a9d54ff7ecefa7feb7c071111e7f6f17a14117b3b", 141124),
    ("99cf70803d3959cb668b04e1ab76173901d1bc9cee02f083f0301dcb2b2e9a36", 207251),
    ("4b40f9b63540fe92653d1d824c875fbd2f32fea4f00bb40faad663d5427ae40d", 298016),
    ("8d197eba27db5b16451b8c64fc76e1010ecef914593c5fa60292a51cdbbc6ebd", 189298),
    ("f8f010ededfd730cc1031d6d2727e18ecf5a73cc15f4978ba89c71f047f522cc", 198110),
    ("ab88a467c1446e2594e7fe5f5a037a55daeed2f1a5215cc895686f1106712007", 162616),
    ("4d737f9768fa5859c0d1acd3402445127c2b09b29edefda03d0ff016a56bca41", 179169),
    ("6c586193df0f623cc83fb198ab0326497d3735cc3f9594a1fdd972dfb6910204", 188270),
    ("726ea9953163b1316f2addc552d0b3be2ff6158d4366dbe72c094dac70fc9960", 285044),
    ("9757c2ac25f906352a4d04c88e258efc8a57f2fcc5f8076858f5f45c736cf020", 292279),
    ("2a3bbbc58bd7f64045b166ddb520bc35b48ed752dc98d60a20365c5c1c5416a5", 269128),
    ("11fd9dea9018b51f25591e6587c4472e24e8212839058558bd95db3d95914901", 271153),
    ("92cff321d4e1428736a4af8946f97d238f4e6637871ce17eca2231a4abf61c15", 219822),
    ("2b740fd72c670203744387f894c10a8b1cbfddeb5d75ce701bbb3263049e89b9", 233045),
    ("434e57937632c027b9ec538ce0382b1cde57bf65f276a4ef3020f0a0c39d4602", 208082),
    ("35e5b0417987624a5744179ef150fdfefc60085533601c16c06bc899151e3e09", 240977),
    ("1722e092428293144d16fb8e5b2bef157f10896ea9a1383efc74108775e69819", 161509),
    ("5c84ccd573bf86abb51bcdcf7c117b341a0c57e7bf379df4dc5e66533aeed369", 162813),
    ("4c1bc000d7ad31d582dda9c6d97840503fe249cfe6ef68f5fc3a04dcb42de442", 245529),
    ("90e982fb75808310cf7b2e7b3d7f58dfab5ac4fa0ba25eaa8d6aa6f55ee06ed3", 315733),
    ("25e352aa4f95aed7b722fcb212d1a286b284ad5ea80a0846b4d190b846bb3791", 168369),
    ("10e8ad20feb78404e797ef503eb30325eb3011d2392e1c68028563505ee76c3a", 143568),
    ("b039aa6dd571c8c9644ceae810126feadf2a33d4a85d8209aadb1691108f25dd", 218132),
    ("ab7e8a7db60c3f48f3eaac98ac8a72e4faf027feff5ece6a4b03d9edef1472ca", 177571),
    ("d1e72f8900471dff7418db49cc1670dfb3e5274237056bbc7fd14759c6ea014a", 293586),
    ("e2fb38541feb7ea28f6abcd24a1e3dadefb486cdc3b4dc162a9ae8307dcfc2e5", 275866),
    ("c2e283191d6eaab101c1daa065055846611dc30b72f517259d9bb32f6513b061", 307445),
    ("07cfb2234a5fc4726982e19995af8cb41b251ce568a3831041d8971548a37034", 180874),
    ("27b52ccbb5d59162af31eba5e3365a9d347a15893665dac2b2e6d1b3175a376a", 229986),
    ("df24388f40487b374263386d4975a95c2d89d1629000343f499d1522cee41a6c", 170607),
    )
    poses = tuple(
        Artifact(
            path=f"data/sketchfab/dress1/{name}",
            sha256="421cc64bab8e8b33ce2cd84cf716cfe80276ec4f858550c891f17b727fa4975c",
            size=86646,
            group="demo",
            scene="dress",
            note="configs/editing/dress_*.yml dataset.path (camera poses)",
        )
        for name in ("transforms_train.json", "transforms_test.json")
    )
    renders = tuple(
        Artifact(
            path=f"data/sketchfab/dress1/r_{i}.png",
            sha256=digest,
            size=size,
            group="demo",
            scene="dress",
            note="configs/editing/dress_*.yml dataset.path (800x800 RGBA renders)",
        )
        for i, (digest, size) in enumerate(frames)
    )
    return poses + renders


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        "checkpoints/nerfsyn/chair/model.pth",
        "44bf0ed53f3dccf8d0fb5168b70847ca23d820fbdecb2b27f37f64db6cad3230",
        143020180,
        "checkpoints",
        "chair",
        "configs/nerfsyn/chair.yml test.load_path (use_sh: true, hence the size)",
    ),
    Artifact(
        "checkpoints/nerfsyn/drums/model.pth",
        "6d7a04701ebe02ab8296f1b39167b3a595c1f22c1a02330900d6479680587ab6",
        34016596,
        "checkpoints",
        "drums",
        "configs/nerfsyn/drums.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/ficus/model.pth",
        "eb75983abc7f8daf44b93d3bad1c4cfa7d524126eb79e461cac02c3be0f81278",
        34012372,
        "checkpoints",
        "ficus",
        "configs/nerfsyn/ficus.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/lego/model.pth",
        "175ef2ce5863cb487f2a6922e570e08095b8b030ef725c52646f13f6eda8d687",
        34102228,
        "checkpoints",
        "lego",
        "configs/nerfsyn/lego.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/materials/model.pth",
        "0489b2ac2b36a3f07b99e760bd1b1262e7565cdb977e73b92b9c0091940cd79d",
        24826836,
        "checkpoints",
        "materials",
        "configs/nerfsyn/materials.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/mic/model.pth",
        "f52c1257465882ec9acf17ae2cb026a28336242d539d93d36f15177ab09517f4",
        34129108,
        "checkpoints",
        "mic",
        "configs/nerfsyn/mic.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/ship/model.pth",
        "92fcf985060907d980088ee12b71386ba19a375b6d6c5ec52edcc25e4dfbd7de",
        34352276,
        "checkpoints",
        "ship",
        "configs/nerfsyn/ship.yml test.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/lego/base.pth",
        "424fb65ba379baa34c11597bdcf3b9b1873ca1feb09378bbaaf8a97ffe437991",
        34037838,
        "checkpoints",
        "lego",
        "configs/nerfsyn/lego.yml load_path (stage-one weights for the -ft run)",
    ),
    Artifact(
        "checkpoints/nerfsyn/materials/base.pth",
        "dfb880857ff91fef4c0426a86a839957f32ad5a1ab92a1e031406879632226d5",
        24831758,
        "checkpoints",
        "materials",
        "configs/nerfsyn/materials.yml load_path (stage-one weights for the -ft run)",
    ),
    Artifact(
        "checkpoints/nerfsyn/mic/base.pth",
        "693219c554c139536b731524185804c12d0e1a10aeb3968b982f330385da9ec7",
        34271374,
        "checkpoints",
        "mic",
        "configs/nerfsyn/mic.yml load_path (stage-one weights for the -ft run)",
    ),
    Artifact(
        "checkpoints/nerfsyn/ship/base.pth",
        "97f97a125a884ab8cb43edc37405b66f27dea7a034ac3c43b7638202b55d4ecb",
        34250190,
        "checkpoints",
        "ship",
        "configs/nerfsyn/ship.yml load_path (stage-one weights for the -ft run)",
    ),
    Artifact(
        "checkpoints/nerfsyn/init/chair.ply",
        "203ab09e3ba4c82fd07a8d86955a0fe42a2ee02e975e65df8a832d1655971bcf",
        810233,
        "checkpoints",
        "chair",
        "configs/nerfsyn/chair.yml geoms.points.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/init/lego.ply",
        "03c409c27ff69c1fcadcff677ed5bf6a66e7c745c093d6a241c08b0aeb46fdd0",
        810233,
        "checkpoints",
        "lego",
        "configs/nerfsyn/lego.yml geoms.points.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/init/materials.ply",
        "6ec5ad26e9bda3a6948d17870483914fa83efd0eea617230a30d3afd62e8501b",
        810233,
        "checkpoints",
        "materials",
        "configs/nerfsyn/materials.yml geoms.points.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/init/mic.ply",
        "545fac755d99d6029a76002c76179446964b73e7452f4f792776306743fab4eb",
        810233,
        "checkpoints",
        "mic",
        "configs/nerfsyn/mic.yml geoms.points.load_path",
    ),
    Artifact(
        "checkpoints/nerfsyn/init/ship.ply",
        "561304e0b9777bcd2b6ba113d1de44a73dbe686d3895723b05e6a7213a31e243",
        810233,
        "checkpoints",
        "ship",
        "configs/nerfsyn/ship.yml geoms.points.load_path",
    ),
    Artifact(
        "demo/dress/papr.pth",
        "24d58a140084a9eeeac5a0a893f6171bc86a24994a804bb265036f2fec22b0d6",
        3802153,
        "demo",
        "dress",
        "configs/editing/dress_papr.yml test.load_path; dress_uv.yml load_path",
    ),
    Artifact(
        "demo/dress/control_points.ply",
        "102969f25ae849a082e5e6fd2a2918bbe2b4762bfac3d2613a746c10248758f0",
        360148,
        "demo",
        "dress",
        "both editing configs: original_control_points_path + geoms.points.load_path",
    ),
    Artifact(
        "demo/dress/nuvo_model_with_texture.ckpt",
        "e64426cb622782c526d10eec002cbadac2406a925ce87c09bcb55052e5c740f9",
        11147349,
        "demo",
        "dress",
        "stage-b atlas + texture, the input to render_edit.py",
    ),
    Artifact(
        "demo/dress/texture_map_grid.png",
        "a47fe365d4febe19a3d9a65e8cba3b79141cc9448c107101ab6c36ac58dffc58",
        653792,
        "demo",
        "dress",
        "learned 4-chart grid, 1024x256; render_edit.py --texture_is_grid",
    ),
    *_dress_frames(),
    *_dress_dataset(),
)

GROUPS = ("checkpoints", "demo")


#: The credit CC BY 4.0 requires, as the exact bytes written to every ``NOTICE`` in
#: :data:`WRITTEN`.  Keeping it in the source is what lets the file be verified: its sha256 is
#: computed from this literal at import, so ``--verify`` catches a NOTICE that was edited or
#: truncated the same way it catches a corrupt checkpoint.
DRESS_NOTICE = """\
Third-party asset credit -- PointGT dress editing demo
======================================================

These files are derived from a third-party 3D asset:

    "Dress with gold leaves" by Canvastique3D
    https://sketchfab.com/canvastique3d

    Licensed under the Creative Commons Attribution 4.0 International licence
    (CC BY 4.0): http://creativecommons.org/licenses/by/4.0/

Everything the PointGT release ships for this scene is a derived work of that
asset:

  demo/dress/control_points.ply          point cloud sampled from its mesh
  demo/dress/rbf_pcds/edit_*.ply         deformations of that point cloud
  demo/dress/papr.pth                    renderer fitted to its renders
  demo/dress/nuvo_model_with_texture.ckpt   UV atlas + texture fitted to the same
  demo/dress/texture_map_grid.png        texture the atlas learned, packed as a
                                         chart grid
  data/sketchfab/dress1/r_*.png          renders of the asset (800x800 RGBA)
  data/sketchfab/dress1/transforms_*.json   the camera poses those were rendered
                                         from

CC BY 4.0 permits redistribution and modification, commercially included, on
condition that the author is credited and that changes are indicated. The
changes made here: the asset was rendered from 100 viewpoints, resampled and
deformed as point clouds, and fitted by the models in this repository.

If you redistribute any of these files, keep this notice with them.
"""

#: Where it is written.  ``demo/dress/`` holds the checkpoints and clouds; ``data/sketchfab/dress1``
#: holds the renders, and is fetched separately by ``--what demo``, so the credit has to sit in both
#: or one of them travels without it.
NOTICE_PATHS = ("demo/dress/NOTICE", "data/sketchfab/dress1/NOTICE")

_NOTICE_BYTES = DRESS_NOTICE.encode("utf-8")

#: The exact bytes each written path receives, keyed by path so a second written file cannot
#: silently inherit the first one's content.
WRITTEN_CONTENT = {path: _NOTICE_BYTES for path in NOTICE_PATHS}

#: Files this script *writes* rather than fetches.  A licence credit has no upstream copy to fetch:
#: there is no archived byte sequence to pin, so shipping it in :data:`ARTIFACTS` would have meant
#: downloading the one file in the release that must be right, unverified.  Authored above instead,
#: it is still pinned -- the sha256 is measured on the literal, so a NOTICE that was edited or
#: truncated fails ``--verify`` exactly like a corrupt checkpoint.
WRITTEN = tuple(
    Artifact(
        path=path,
        sha256=hashlib.sha256(_NOTICE_BYTES).hexdigest(),
        size=len(_NOTICE_BYTES),
        group="demo",
        scene="dress",
        note="CC BY 4.0 credit for the dress asset -- written by download.py, not fetched",
    )
    for path in NOTICE_PATHS
)


def sha256_of(path: Path) -> str:
    """Streaming sha256 so a 143 MB checkpoint never lands in memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter(artifacts: Iterable[Artifact], groups: set[str], scene: str | None) -> list[Artifact]:
    chosen = [a for a in artifacts if a.group in groups]
    if scene is not None:
        chosen = [a for a in chosen if a.scene == scene]
    return chosen


def select(what: Sequence[str], scene: str | None) -> tuple[list[Artifact], list[Artifact]]:
    """Filter both registries by ``--what`` group and ``--scene``.

    Returns ``(fetched, written)``: the artifacts to download, and the ones :data:`WRITTEN` says
    this script authors locally. Every command handles both, so a ``--what``/``--scene`` that
    selects the dress also gets its licence credit.
    """
    groups = set(GROUPS) if (not what or "all" in what) else set(what)
    fetched = _filter(ARTIFACTS, groups, scene)
    written = _filter(WRITTEN, groups, scene)
    if scene is not None and not fetched and not written:
        known = sorted({a.scene for a in ARTIFACTS})
        raise SystemExit(f"no artifacts for scene {scene!r}; known scenes: {', '.join(known)}")
    return fetched, written


def human(size: int) -> str:
    """Sizes for the table; the registry stores exact bytes, this is display only."""
    if size <= 0:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}GB"


OK, STALE, ABSENT, UNPINNED = "ok", "sha256 mismatch", "missing", "present (unpinned)"


def check(local: Path, art: Artifact) -> str:
    """Classify what is on disk for ``art`` without touching the network."""
    if not local.is_file():
        return ABSENT
    if art.pending:
        return UNPINNED
    return OK if sha256_of(local) == art.sha256 else STALE


def write_local(art: Artifact, dest: Path) -> None:
    """Write one :data:`WRITTEN` artifact from its literal.

    Bytes, not text: the pinned hash is of the literal encoded as UTF-8 with LF endings, and a
    platform that rewrote those to CRLF on the way out would fail its own ``--verify``.
    """
    local = dest / art.path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(WRITTEN_CONTENT[art.path])


def _hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def _fetch_via_hub(art: Artifact, dest: Path, repo: str, revision: str) -> None:
    """Preferred path: resumable, retried, and honours HF_TOKEN for a gated repo.

    ``local_dir`` puts the file at ``dest / art.path`` with no symlink into the shared cache,
    which is what the verify pass and every ``load_path`` in the configs expect.  The hub's own
    exceptions are multi-paragraph; they are collapsed here so one failure is one line.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        hf_hub_download(
            repo_id=repo,
            filename=art.path,
            repo_type=DEFAULT_REPO_TYPE,
            revision=revision,
            local_dir=str(dest),
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"repo {repo!r} not found. It may not exist yet, or it may be private -- "
            "log in with `hf auth login` or set HF_TOKEN."
        ) from exc
    except RevisionNotFoundError as exc:
        raise RuntimeError(f"revision {revision!r} does not exist in {repo!r}") from exc
    except GatedRepoError as exc:
        raise RuntimeError(f"{repo!r} is gated: accept the terms, then set HF_TOKEN") from exc
    except EntryNotFoundError as exc:
        raise RuntimeError(f"not in {repo!r} @ {revision}: {art.path}") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(str(exc).splitlines()[0]) from exc


def _fetch_via_https(art: Artifact, dest: Path, repo: str, revision: str) -> None:
    """Fallback when ``huggingface_hub`` is absent: stdlib GET of the resolve URL.

    Written to ``<file>.part`` and renamed only on a clean read, so an interrupted transfer can
    never be mistaken for a complete file by the next ``--verify``.
    """
    url = RESOLVE_URL.format(repo=repo, revision=revision, path=art.path)
    local = dest / art.path
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.parent / (local.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pointgt-download/1.0"})
    try:
        with urllib.request.urlopen(request) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle, _CHUNK)
    except urllib.error.HTTPError as exc:
        part.unlink(missing_ok=True)
        hint = ""
        if exc.code in (401, 403):
            hint = "  (repo may be gated: set HF_TOKEN, or install huggingface_hub and log in)"
        raise RuntimeError(f"HTTP {exc.code} for {url}{hint}") from exc
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
    part.replace(local)


def fetch(art: Artifact, dest: Path, repo: str, revision: str, use_hub: bool) -> None:
    """Download one artifact into ``dest`` and leave it at ``dest / art.path``."""
    (dest / art.path).parent.mkdir(parents=True, exist_ok=True)
    if use_hub:
        _fetch_via_hub(art, dest, repo, revision)
    else:
        _fetch_via_https(art, dest, repo, revision)


def _print_table(artifacts: list[Artifact], width: int) -> None:
    previous_note = ""
    for art in artifacts:
        digest = art.sha256[:16] if art.sha256 else "PENDING"
        print(f"{art.path.ljust(width)}  {human(art.size):>8}  {digest:<16}  {art.scene}")
        if art.note and art.note != previous_note:
            print(f"{' ' * width}  {' ':>8}  {' ':<16}  {art.note}")
        previous_note = art.note


def cmd_list(chosen: Iterable[Artifact], written: Iterable[Artifact],
             repo: str, revision: str) -> int:
    artifacts, local = list(chosen), list(written)
    print(f"repo   https://huggingface.co/{repo}  (revision {revision})")
    print(f"{len(artifacts)} artifact(s) to fetch, {len(local)} written locally\n")
    width = max((len(a.path) for a in artifacts + local), default=4)
    print(f"{'PATH'.ljust(width)}  {'SIZE':>8}  {'SHA256':<16}  SCENE / NOTE")
    print("-" * (width + 60))
    _print_table(artifacts, width)
    if local:
        print("\n-- written by download.py, not fetched --")
        _print_table(local, width)
    total = sum(a.size for a in artifacts)
    pending = [a.path for a in artifacts + local if a.pending]
    print(f"\ntotal {human(total)} to download")
    if pending:
        print(f"pending (no pinned sha256, downloaded unverified): {', '.join(pending)}")
    return 0


def cmd_verify(chosen: Iterable[Artifact], written: Iterable[Artifact], dest: Path) -> int:
    counts = {OK: 0, STALE: 0, ABSENT: 0, UNPINNED: 0}
    for art in list(chosen) + list(written):
        status = check(dest / art.path, art)
        counts[status] += 1
        print(f"{status:>18}  {art.path}")
    print(
        f"\n{counts[OK]} ok, {counts[STALE]} mismatched, "
        f"{counts[ABSENT]} missing, {counts[UNPINNED]} unpinned"
    )
    return 1 if counts[STALE] else 0


def cmd_download(
    chosen: Iterable[Artifact],
    written: Iterable[Artifact],
    dest: Path,
    repo: str,
    revision: str,
    force: bool,
) -> int:
    use_hub = _hf_available()
    print(f"source  https://huggingface.co/{repo} @ {revision}")
    print(f"via     {'huggingface_hub' if use_hub else 'urllib (huggingface_hub not installed)'}")
    print(f"dest    {dest.resolve()}\n")

    failed: list[str] = []
    skipped = downloaded = 0

    for art in written:
        if not force and check(dest / art.path, art) == OK:
            print(f"     have  {art.path}")
            skipped += 1
            continue
        print(f"    write  {art.path} ({human(art.size)})")
        write_local(art, dest)
        downloaded += 1

    for art in chosen:
        local = dest / art.path
        if not force and check(local, art) == OK:
            print(f"     have  {art.path}")
            skipped += 1
            continue
        if force and local.exists():
            local.unlink()
        print(f"      get  {art.path} ({human(art.size)})", flush=True)
        try:
            fetch(art, dest, repo, revision, use_hub)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the batch
            print(f"     FAIL  {art.path}: {exc}")
            failed.append(art.path)
            continue

        if art.pending:
            print(f"  UNPINNED  {art.path} (no sha256 in the table; not verified)")
            downloaded += 1
            continue
        actual = sha256_of(local)
        if actual != art.sha256:
            local.unlink(missing_ok=True)
            print(f"     FAIL  {art.path}: sha256 {actual[:16]} != {art.sha256[:16]}, deleted")
            failed.append(art.path)
        else:
            downloaded += 1

    print(f"\n{downloaded} downloaded, {skipped} already present, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download.py",
        description="Fetch PointGT checkpoints and demo data from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python download.py --list\n"
            "  python download.py --what checkpoints\n"
            "  python download.py --scene dress\n"
            "  python download.py --verify\n"
        ),
    )
    parser.add_argument(
        "--what",
        action="append",
        choices=("checkpoints", "demo", "all"),
        default=None,
        help="artifact group; repeatable (default: all)",
    )
    parser.add_argument("--scene", default=None, help="restrict to one scene, e.g. lego or dress")
    parser.add_argument(
        "--dest", default=".", help="root that receives checkpoints/, demo/ and data/"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace model repo id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="branch, tag or commit")
    parser.add_argument("--force", action="store_true", help="re-download even if sha256 matches")
    parser.add_argument(
        "--verify", action="store_true", help="check local sha256, download nothing"
    )
    parser.add_argument("--list", action="store_true", help="print the artifact table and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass

    args = build_parser().parse_args(argv)
    what = args.what or ["all"]
    chosen, written = select(what, args.scene)
    dest = Path(args.dest)

    if args.list:
        return cmd_list(chosen, written, args.repo, args.revision)
    if args.verify:
        return cmd_verify(chosen, written, dest)
    return cmd_download(chosen, written, dest, args.repo, args.revision, args.force)


if __name__ == "__main__":
    sys.exit(main())

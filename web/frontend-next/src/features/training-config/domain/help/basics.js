// Facts checked 2026-10-05 against the family registry, caption reader,
// preprocess image filter/task wrapper, and shared preview decode policy.
// Option labels come from the form catalog rather than this content module.
export const FIELD_HELP_BASICS_ZH = {
    model_family: {
        version: 2,
        summary: '选择你要训练的底模所属模型族，让训练使用对应的模型、文本编码和采样流程。',
        recommendation: '按底模权重选择模型族。切换后先核对模型路径、训练方法和缓存，再运行预检。',
        options: [
            { value: 'moss_image', when: '你使用 MOSS-Image 底模权重时。', detail: '首版普通文生图 plain LoRA，冻结文本编码器，需磁盘文本和 latent 缓存。' },
            { value: 'anima', when: '你使用 Anima 底模权重时。' },
            { value: 'krea2_raw', when: '你使用 Krea-2 Raw 底模权重时。' },
            { value: 'z_image', when: '你使用 Z-Image 底模权重时。', detail: '普通 LoRA 已有真机训练验证，其他组合需要确认验证范围。' },
            { value: 'qwen_image_2_1', when: '你使用 Qwen-Image 2.1 底模权重时。', detail: '可用于文生图或编辑训练，具体任务另行选择。' },
        ],
        guidance: [
            {
                title: '切换后检查什么',
                paragraphs: [
                    '检查底模、文本编码器和 VAE 路径是否属于目标模型族，再核对所选训练方法的适用范围。界面允许选择某个网络变体，不代表该组合已经过真机验证。',
                    '切换模型族不会主动下载权重，也不会自动重置全部训练参数。保留下来的编译、块交换和网络配置仍需你重新核对。',
                ],
            },
        ],
        cautions: [
            '模型族必须与底模权重匹配。只修改模型族名称不能把另一种底模或 adapter 权重转换成可用格式。',
            '不要直接沿用其他模型族的文本缓存。图像缓存是否可复用取决于模型族的 VAE 和缓存格式，需要单独确认。',
        ],
        relatedKeys: ['pretrained_model_name_or_path', 'qwen3', 'vae'],
    },
    caption_extension: {
        version: 2,
        summary: '指定与图片同名的文本标注文件扩展名，帮助预处理找到你的文字描述。',
        recommendation: '图片使用同名 .txt 标注时保留 .txt。只有文本标注实际采用其他扩展名时才修改。',
        guidance: [
            {
                title: '怎样填写',
                paragraphs: [
                    '例如 cat.png 对应 cat.txt，填写 .txt。如果标注文件是 cat.caption，则填写 .caption。',
                    '此项只决定文本标注的文件名，不会把 JSON 解释成普通文本标注。使用结构化 JSON 时，应检查“标注来源”。',
                ],
            },
        ],
        cautions: [
            '扩展名与实际文件不一致时，预处理可能读不到文本标注。修改标注内容或来源后，已有文本编码缓存需要重建才能反映新描述。',
        ],
        advanced: [
            {
                title: '与标注来源的关系',
                paragraphs: [
                    '自动来源按目录 captions.json、同名 JSON、同名文本的顺序查找可用标注。查到文本时才使用这里的扩展名。',
                    '明确选择 JSON 来源时使用对应 JSON 读取规则。旧配置的“优先 JSON 标注”仍可能在 JSON 缺失或无效时回退到文本。',
                ],
            },
        ],
        relatedKeys: ['caption_source_mode', 'prefer_json_caption'],
    },
    drop_lowres_images: {
        version: 2,
        summary: '在预处理时跳过原始像素总数低于“最低像素数”的图片。',
        recommendation: '一般保留开启，并先检查被跳过的图片。若小图包含你必须保留的主体或风格，可降低门槛或关闭过滤。',
        options: [
            { value: 'true', when: '你希望预处理排除低于门槛的原始图片时。', detail: '具体门槛由“最低像素数”决定，页面默认是 500000。' },
            { value: 'false', when: '你需要保留小图，或希望自己筛选数据时。', detail: '预处理不再按最低像素数跳过图片。' },
        ],
        guidance: [
            {
                title: '怎样判断图片太小',
                paragraphs: [
                    '门槛比较的是原始宽度 × 高度，不是某一边的长度。在 500000 像素门槛下，512×512 会被跳过，1024×512 会保留。',
                    '训练分辨率决定后续缩放和分桶，不是这里的过滤门槛。放大小图也不会增加原始细节。',
                ],
            },
        ],
        cautions: [
            '这是预处理过滤开关，不是训练中的实时开关。修改它不会自动清理或重新生成已有预处理结果。',
        ],
        relatedKeys: ['min_pixels', 'resolution'],
    },
    min_pixels: {
        version: 2,
        summary: '设置预处理允许保留的原始图片最低像素总数，按宽度 × 高度计算。',
        recommendation: '开启低分辨率过滤时，可先用页面默认的 500000 检查数据。要保留更小的图片就降低门槛，要排除更多小图就提高门槛。',
        guidance: [
            {
                title: '如何选择门槛',
                paragraphs: [
                    '只有像素总数严格低于门槛的图片才会被跳过，等于门槛的图片会保留。500000 是像素总数，不要求图片为正方形。',
                    '例如 512×512 是 262144 像素，1024×512 是 524288 像素。在默认门槛下，前者跳过，后者保留。',
                    '降低门槛能保留更多小图，但这些图片可能需要放大。提高门槛会减少可用图片，先检查是否丢掉重要样本。',
                ],
            },
        ],
        cautions: [
            '关闭“过滤低分辨率图”后，预处理会使用 0 作为过滤门槛，此处数值不再用于过滤。门槛为 0 时也不按像素总数过滤。',
            '此项不决定训练分辨率，也不会在训练过程中重新筛选已有数据。调整后需要重新执行相应预处理才能应用新门槛。',
        ],
        relatedKeys: ['drop_lowres_images', 'resolution'],
    },
    sample_vae_decode: {
        version: 2,
        summary: '选择训练预览图的解码方式。VAE（图像编解码模型）把生成的中间结果还原成图片。',
        recommendation: '通常保留页面默认的分块 GPU 解码。若该阶段仍缺显存，可选 CPU 并先确认主机内存余量，或用自动模式允许逐级回退。',
        options: [
            {
                value: 'auto',
                when: '你希望先按 VAE 当前设置在 GPU 解码，并在明确的 CUDA 显存不足时回退。',
                detail: '依次尝试常规 GPU、分块 GPU、分块 CPU。常规阶段保留 VAE 原有分块设置，不保证一定整图解码。',
            },
            {
                value: 'tiled',
                when: '你希望直接分块 GPU 解码，降低预览解码的显存峰值时。',
                detail: '分块可能增加耗时，显存不足时不会自动改用 CPU。',
            },
            {
                value: 'cpu',
                when: '你希望把预览 VAE 解码移到 CPU，并且主机内存有足够余量时。',
                detail: 'CPU 解码也启用分块，并先检查主机内存预算。通常需要等待更久。',
            },
        ],
        cautions: [
            '此项只影响训练预览的 VAE 解码，不处理训练前向或反向传播的显存不足，也不会重启训练。',
            '自动模式只对明确的 CUDA 显存不足逐级回退，其他错误不会通过换设备自动解决。CPU 内存预算检查通过也不保证实际解码一定成功。',
        ],
        advanced: [
            {
                title: '解码与兼容配置',
                paragraphs: [
                    'latent 是模型生成的压缩图像表示，VAE 解码负责把它转换成可查看的图片。分块解码把这一阶段拆成较小区域，以控制内存占用。',
                    '全部五个注册模型族共用这套预览解码策略。旧键 qwen_sample_vae_decode 仍作为兼容入口，可选配置值为 auto、tiled、cpu。',
                ],
            },
        ],
        relatedKeys: ['sample_prompts', 'sample_every_n_steps'],
    },
};

class PipelineStage:
    """Pipeline 插件基类"""
    
    def process(self, data: dict) -> dict:
        """
        处理管道节点。
        :param data: 流经管道字典，包含 'frame' 等上下文
        :return: 增加/过滤属性后的 context 字典
        """
        raise NotImplementedError

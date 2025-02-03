import yaml

# カスタムDumperを作成して、特定の条件でフォーマットを調整
class CustomDumper(yaml.Dumper):
    def represent_list(self, data:list):
        # リストはFlowスタイル（[ ]を使用）で表示
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def Custum(data: dict, output_file):
    yaml.add_representer(list, CustomDumper.represent_list, Dumper=CustomDumper)

    # if output_file is None or output_file == "":
    #     output_file = os.getcwd()
    # YAMLをファイルに出力
    with open(output_file, "w", encoding="utf-8") as file:
        yaml.dump(data, file, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)

    print(f"YAMLデータを '{output_file}' に保存しました。")
import pyttsx3
import os
import time


def initialize_tts():
    """初始化文本转语音引擎"""
    try:
        engine = pyttsx3.init()
        # 设置语音参数
        engine.setProperty('rate', 150)  # 语速
        engine.setProperty('volume', 0.9)  # 音量 (0-1)

        # 获取可用的语音列表（调试用）
        voices = engine.getProperty('voices')
        print("可用的语音:", [voice.name for voice in voices])

        return engine
    except Exception as e:
        print(f"初始化语音引擎失败: {e}")
        return None


def save_and_play_letter(letter, engine):
    """保存并播放字母发音"""
    letter = letter.upper()
    filename = f"{letter}.wav"

    print(f"\n处理字母: {letter}")

    try:
        # 直接保存为WAV文件
        print(f"正在保存到 {filename}...")
        engine.save_to_file(letter, filename)
        engine.runAndWait()  # 必须调用这个才能实际保存文件

        # 检查文件是否生成
        if os.path.exists(filename):
            print(f"成功保存: {filename} (大小: {os.path.getsize(filename) / 1024:.1f} KB)")
        else:
            print("文件保存失败！")
            return

        # 播放音频
        print("正在播放...")
        engine.say(letter)
        engine.runAndWait()

    except Exception as e:
        print(f"处理字母 {letter} 时出错: {e}")


def main():
    print("字母发音程序 - Windows版")
    print("输入字母将播放发音并保存为WAV文件")
    print("输入 'quit' 退出程序\n")

    engine = initialize_tts()
    if not engine:
        print("无法初始化语音引擎，请检查系统是否支持")
        return

    # while True:
    #     user_input = input("请输入一个字母 (a-z): ").strip().lower()
    for i in range(26):
        user_input = chr(ord('A') + i)

        if not user_input:
            continue

        if user_input == 'quit':
            print("退出程序...")
            break

        if len(user_input) != 1 or not user_input.isalpha():
            print("请输入单个字母 (a-z)!")
            continue

        save_and_play_letter(user_input, engine)

        # 短暂暂停
        time.sleep(0.3)


if __name__ == "__main__":
    main()

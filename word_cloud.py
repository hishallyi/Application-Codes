"""
@Author : Hishallyi
@Date   : 2024/7/6
@Code   : 词云
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 生成的文本
text = """
在一个阳光明媚的早晨，小镇上的人们忙碌地开始了一天的生活。孩子们背着书包走向学校，商贩们在市场上热闹地叫卖，老人们在公园里悠闲地散步。小镇虽然不大，但却充满了温馨与活力。

小镇中心有一座古老的钟楼，每到整点，钟声便会响起，提醒人们时间的流逝。钟楼旁边是一条小河，清澈的河水缓缓流淌，两岸种满了各色的花草，春天时节，繁花似锦，景色宜人。河边的小路上常常能看到散步的情侣、跑步的年轻人和骑自行车的小孩。每个人都在享受着这片宁静与美好。

镇上有一所历史悠久的小学，学校的老师们都非常敬业，他们不仅传授知识，还教孩子们如何做人。学生们在课堂上认真听讲，课间嬉戏玩耍，校园里充满了欢声笑语。放学后，孩子们三五成群地结伴回家，路上聊着学校里的趣事，脸上洋溢着幸福的笑容。

市场是小镇最热闹的地方，各种新鲜的蔬菜、水果、鱼肉应有尽有，商贩们热情地招呼顾客，讨价还价的声音此起彼伏。市场不仅是购物的地方，也是人们交流信息、增进感情的场所。每到周末，市场上更是人头攒动，大家在这里购物、交流，享受着生活的乐趣。

小镇的夜晚同样迷人，街道两旁的路灯散发出柔和的光芒，商店的橱窗里灯火通明，照亮了整个街区。人们晚饭后喜欢出来散步，享受夜晚的凉风。年轻人聚集在咖啡馆里聊天，老人们则在广场上跳舞、唱歌，整个小镇沉浸在一片祥和与欢乐之中。

这个小镇虽然没有大城市的繁华，却有着独特的魅力。这里的人们热情友善，生活节奏缓慢，每个人都能找到属于自己的幸福和快乐。
"""

# 生成词云
wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400, background_color='white').generate(text)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
queries_history = dict()


class Ask:
    def __init__(self, choices=['y', 'n']):
        self.choices = choices

    def ask(self):
        if max([len(x) for x in self.choices]) > 1:
            for i, x in enumerate(self.choices):
                print("{0}. {1}".format(i, x))
            x = int(input())
            return self.choices[x]
        else:
            ans = None
            while ans not in self.choices:
                print("/".join(self.choices))
                ans = input()
            queries_history[list(queries_history.keys())[-1]] = ans
            return ans


class Content:
    def __init__(self, x):
        self.x = x


class If(Content):
    pass


class AND(Content):
    pass


class OR(Content):
    pass


class NoMoreUnivers(Exception):
    pass


rules = {
    'default': Ask(['y', 'n']),
    'хорошее математическое образование': If(AND(['готов изучать математику долго и упорно', 'любишь математику'])),
    'хорошее гуманитарное образование': If(
        OR([AND(['готов долго и упорно изучать гуманитарные науки', 'любишь рассуждать']), 'любишь гуманитарные науки',
            'стал призером олимпиады по гуманитарному предмету и хочешь развиваться в этой области'])),
    'хорошее медицинское образование': If(
        AND(['не боишься крови, операций, трупов и т.п.', 'хочешь лечить людей',
             'готов к длительному и непростому образовательному процессу', ])),
    'хорошее техническое образование': If(
        OR(['любишь физику и разбираешься в ней', 'нравится работать с различной техникой'])),
    'хорошее педагогическое образование': If(AND(['имеешь преподавательские задатки', 'хочешь преподавать (учить)'])),
    'хорошее IT-образование': If(AND([OR(['обладаешь алгоритмическим мышлением', 'продвинутый пользователь ПК']),
                                      'готов уделять много времени самообучению', 'нравится информатика'])),
    'хорошее химико-биологическое образование': If(
        AND(['любишь биологию и разбираешься в ней', 'любишь химию и разбираешься в ней'])),
    'академическое образование': If('готов изучать много предметов для общего развития'),
    'практическое образование': If(
        AND(['готов много практиковаться по специальности', 'не хочешь изучать много предметов для общего развития'])),
    'неопределенность в профессии': If(
        OR(['не выбрал профессию', 'не готов работать по этой профессии большую часть жизни',
            'не нравится выбранная профессия'])),
    'хорошее юридическое образование': If(AND(['нравится изучать законы', 'нравится право'])),
    'хорошее строительное образование': If(AND(['любишь архитектуру', 'любишь физику и разбираешься в ней'])),
    'хорошее экономическое образование': If(
        OR(['мечтаешь о собственном деле', 'собираешься работать в сфере управления',
            'хочешь изучать экономическое процессы'])),
    'нет военной кафедры': If('не нужна военная кафедра'),
    'европейская система образования (выбор предметов)': If('не хочешь иметь строго фиксированный учебный план'),
    'госслужба': If(
        AND(['готов изучать политику', 'нравится изучать законы', 'хочешь заниматься благоустройством страны',
             'хочешь изучать экономическое процессы'])),
    'погружение в науку': If(
        OR([AND(['способен работать в команде', 'хочешь двигать науку вперед']), 'имеешь собственные научные наработки',
            'мне неважно развитие в науке в ВУЗе', 'хочешь заниматься серьёзной наукой'])),
    'международное взаимодействие': If(
        OR(['большое количество иностранных студентов', 'есть программа международной мобильности',
            'есть совместные программы с зарубежными ВУЗами', 'мне не важно международное взаимодействие в ВУЗе'])),
    'добротное изучение иностранных языков': If(
        OR(['хочешь изучать иностранные языки', 'хочешь улучшить знание иностранного языка'])),
    'внеучебные активности': If(
        OR(['в ВУЗе должны быть волонтёрские центры', 'неважно, есть ли у ВУЗа внеучебные активности',
            'хочу посещать спортивные секции в ВУЗе', 'хочу участвовать в конкурсах и концертах от ВУЗа',
            'хочу ходить в походы или на экскурсии от ВУЗа'])),
    'заряженные на успех студенты': If(OR(['мне неважно моё окружение в ВУЗе',
                                           'не менее 80% выпускников ВУЗа должно быть трудоустроено в первый год после выпуска',
                                           'уровень зарплат выпускников ВУЗа должен быть выше среднего',
                                           'хочу учиться в очень конкурентной среде'])),
    'престиж': If(OR(['мне неважна престижность ВУЗа', 'хочешь учиться в престижном ВУЗе'])),
    'ВУЗ:МГУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                           'хорошее IT-образование', 'хорошее гуманитарное образование',
                           'хорошее математическое образование', 'хорошее педагогическое образование',
                           'хорошее техническое образование', 'хорошее химико-биологическое образование',
                           'хорошее экономическое образование', 'хорошее юридическое образование']),
                       'академическое образование', 'внеучебные активности', 'заряженные на успех студенты',
                       'погружение в науку', 'престиж'])),
    'ВУЗ:ВШЭ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                           'хорошее IT-образование', 'хорошее гуманитарное образование',
                           'хорошее математическое образование', 'хорошее техническое образование',
                           'хорошее экономическое образование']), 'внеучебные активности',
                       'европейская система образования (выбор предметов)', 'заряженные на успех студенты',
                       'международное взаимодействие', 'погружение в науку', 'престиж'])),
    'ВУЗ:КФУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                           'хорошее IT-образование', 'хорошее гуманитарное образование',
                           'хорошее математическое образование', 'хорошее педагогическое образование',
                           'хорошее техническое образование', 'хорошее химико-биологическое образование',
                           'хорошее экономическое образование', 'хорошее юридическое образование']),
                       'академическое образование', 'внеучебные активности', 'международное взаимодействие',
                       'нет военной кафедры', 'престиж'])),
    'ВУЗ:МФТИ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                            'хорошее математическое образование', 'хорошее техническое образование',
                            'хорошее химико-биологическое образование']),
                        'внеучебные активности', 'заряженные на успех студенты', 'международное взаимодействие',
                        'погружение в науку', 'престиж'])),
    'ВУЗ:МГМУ им. Сеченова': If(
        AND([OR(['хорошее медицинское образование', 'хорошее химико-биологическое образование']),
             'внеучебные активности', 'заряженные на успех студенты', 'международное взаимодействие',
             'практическое образование', 'престиж'])),
    'ВУЗ:МТУСИ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                             'хорошее математическое образование', 'хорошее техническое образование']),
                         'внеучебные активности', 'практическое образование'])),
    'ВУЗ:КФУ им. Вернадского': If(AND([OR(['хорошее гуманитарное образование', 'хорошее техническое образование',
                                           'хорошее экономическое образование', 'хорошее юридическое образование']),
                                       'внеучебные активности', 'заряженные на успех студенты',
                                       'международное взаимодействие'])),
    'ВУЗ:РАНХиГС': If(AND([OR(['хорошее IT-образование', 'хорошее гуманитарное образование',
                               'хорошее экономическое образование', 'хорошее юридическое образование']), 'госслужба',
                           'заряженные на успех студенты', 'международное взаимодействие', 'престиж'])),
    'ВУЗ:УРФУ им. Ельцина': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                                        'хорошее IT-образование', 'хорошее гуманитарное образование',
                                        'хорошее математическое образование', 'хорошее техническое образование',
                                        'хорошее химико-биологическое образование',
                                        'хорошее экономическое образование']),
                                    'внеучебные активности', 'заряженные на успех студенты',
                                    'международное взаимодействие'])),
    'ВУЗ:МАИ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                           'хорошее математическое образование', 'хорошее техническое образование']),
                       'внеучебные активности', 'европейская система образования (выбор предметов)',
                       'заряженные на успех студенты', 'международное взаимодействие', 'погружение в науку',
                       'престиж'])),
    'ВУЗ:ТГУ': If(AND([OR(['хорошее гуманитарное образование', 'хорошее педагогическое образование',
                           'хорошее экономическое образование', 'хорошее юридическое образование']),
                       'академическое образование', 'международное взаимодействие', 'погружение в науку'])),
    'ВУЗ:ТПУ': If(AND([OR(['хорошее IT-образование', 'хорошее математическое образование',
                           'хорошее техническое образование', 'хорошее химико-биологическое образование']),
                       'внеучебные активности', 'заряженные на успех студенты', 'погружение в науку',
                       'практическое образование', 'престиж'])),
    'ВУЗ:НГУ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                           'хорошее математическое образование', 'хорошее техническое образование',
                           'хорошее химико-биологическое образование', 'хорошее экономическое образование']),
                       'международное взаимодействие', 'нет военной кафедры',
                       'погружение в науку'])),
    'ВУЗ:СПбГУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                             'хорошее IT-образование', 'хорошее гуманитарное образование',
                             'хорошее математическое образование', 'хорошее педагогическое образование',
                             'хорошее техническое образование', 'хорошее экономическое образование',
                             'хорошее юридическое образование']), 'академическое образование',
                         'заряженные на успех студенты', 'международное взаимодействие', 'престиж'])),
    'ВУЗ:СПбПУ': If(AND([OR(['хорошее IT-образование', 'хорошее математическое образование',
                             'хорошее строительное образование', 'хорошее техническое образование',
                             'хорошее экономическое образование']), 'заряженные на успех студенты',
                         'погружение в науку', 'престиж'])),
    'ВУЗ:МИФИ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                            'хорошее IT-образование', 'хорошее математическое образование',
                            'хорошее техническое образование', 'хорошее экономическое образование']),
                        'академическое образование', 'заряженные на успех студенты', 'международное взаимодействие',
                        'погружение в науку', 'престиж'])),
    'ВУЗ:МГТУ им. Баумана': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                                        'хорошее математическое образование', 'хорошее техническое образование']),
                                    'внеучебные активности', 'заряженные на успех студенты',
                                    'международное взаимодействие', 'погружение в науку', 'престиж'])),
    'ВУЗ:МГИМО': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                             'хорошее гуманитарное образование', 'хорошее педагогическое образование',
                             'хорошее экономическое образование', 'хорошее юридическое образование']), 'госслужба',
                         'заряженные на успех студенты', 'международное взаимодействие', 'престиж'])),
    'ВУЗ:ИТМО': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование',
                            'хорошее математическое образование', 'хорошее техническое образование',
                            'хорошее химико-биологическое образование', 'хорошее экономическое образование']),
                        'внеучебные активности', 'заряженные на успех студенты', 'международное взаимодействие',
                        'погружение в науку', 'практическое образование', 'престиж'])),
    'ВУЗ:ФУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                          'хорошее гуманитарное образование', 'хорошее экономическое образование',
                          'хорошее юридическое образование']), 'академическое образование',
                      'заряженные на успех студенты', 'международное взаимодействие', 'престиж'])),
    'ВУЗ:РЭУ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование', 'хорошее гуманитарное образование',
                           'хорошее экономическое образование', 'хорошее юридическое образование']),
                       'внеучебные активности', 'заряженные на успех студенты', 'международное взаимодействие',
                       'нет военной кафедры'])),
    'ВУЗ:МИСиС': If(AND([OR(['добротное изучение иностранных языков', 'хорошее IT-образование',
                             'хорошее математическое образование', 'хорошее техническое образование',
                             'хорошее химико-биологическое образование']), 'внеучебные активности',
                         'заряженные на успех студенты', 'международное взаимодействие', 'нет военной кафедры',
                         'погружение в науку', 'практическое образование', 'престиж'])),
    'ВУЗ:РУДН': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                            'хорошее IT-образование', 'хорошее гуманитарное образование',
                            'хорошее математическое образование', 'хорошее медицинское образование',
                            'хорошее педагогическое образование', 'хорошее экономическое образование']),
                        'внеучебные активности', 'заряженные на успех студенты', 'международное взаимодействие',
                        'нет военной кафедры'])),
    'ВУЗ:РНИМУ им. Пирогова': If(
        AND([OR(['хорошее медицинское образование', 'хорошее химико-биологическое образование']),
             'заряженные на успех студенты', 'международное взаимодействие', 'нет военной кафедры',
             'погружение в науку', 'практическое образование', 'престиж'])),
    'ВУЗ:КГМУ': If(AND([OR(['хорошее медицинское образование', 'хорошее химико-биологическое образование']),
                        'заряженные на успех студенты', 'международное взаимодействие', 'нет военной кафедры',
                        'практическое образование'])),
    'ВУЗ:МЭИ': If(
        AND([OR(['хорошее IT-образование', 'хорошее математическое образование', 'хорошее техническое образование']),
             'внеучебные активности', 'заряженные на успех студенты', 'погружение в науку', 'практическое образование',
             'престиж'])),
    'ВУЗ:СФУ': If(AND([OR(['неопределенность в профессии', 'хорошее IT-образование', 'хорошее гуманитарное образование',
                           'хорошее математическое образование', 'хорошее техническое образование']),
                       'внеучебные активности', 'заряженные на успех студенты', 'практическое образование'])),
    'ВУЗ:ДВФУ': If(AND([OR(['неопределенность в профессии', 'хорошее гуманитарное образование',
                            'хорошее математическое образование', 'хорошее техническое образование']),
                        'академическое образование', 'внеучебные активности', 'заряженные на успех студенты',
                        'погружение в науку', 'престиж'])),
    'ВУЗ:РГУ им. Губкина': If(AND([OR(['хорошее строительное образование', 'хорошее техническое образование',
                                       'хорошее химико-биологическое образование']), 'академическое образование',
                                   'заряженные на успех студенты', 'практическое образование', 'престиж'])),
    'ВУЗ:ВАВT': If(AND([OR(['хорошее гуманитарное образование', 'хорошее экономическое образование',
                            'хорошее юридическое образование']), 'внеучебные активности',
                        'международное взаимодействие', 'неопределенность в профессии', 'нет военной кафедры',
                        'практическое образование'])),
    'ВУЗ:МФЮА': If(AND([OR(['добротное изучение иностранных языков', 'хорошее математическое образование',
                            'хорошее юридическое образование']),
                        'внеучебные активности', 'нет военной кафедры', 'погружение в науку', 'престиж'])),
    'ВУЗ:МГСУ': If(AND([OR(['хорошее строительное образование', 'хорошее техническое образование']),
                        'неопределенность в профессии', 'нет военной кафедры', 'погружение в науку',
                        'практическое образование'])),
    'ВУЗ:МПГУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                            'хорошее гуманитарное образование', 'хорошее педагогическое образование']),
                        'внеучебные активности', 'нет военной кафедры', 'практическое образование'])),
    'ВУЗ:ЛЭТИ': If(AND([OR(['хорошее IT-образование', 'хорошее гуманитарное образование',
                            'хорошее математическое образование', 'хорошее техническое образование']),
                        'внеучебные активности', 'нет военной кафедры', 'практическое образование'])),
    'ВУЗ:АГУ': If(AND([OR(['неопределенность в профессии', 'хорошее гуманитарное образование',
                           'хорошее педагогическое образование', 'хорошее юридическое образование']),
                       'внеучебные активности', 'нет военной кафедры'])),
    'ВУЗ:РГГУ': If(AND([OR(['добротное изучение иностранных языков', 'неопределенность в профессии',
                            'хорошее гуманитарное образование', 'хорошее юридическое образование']),
                        'внеучебные активности', 'нет военной кафедры', 'погружение в науку', 'престиж'])),
    'ВУЗ:МИРЭА': If(
        AND([OR(['хорошее IT-образование', 'хорошее математическое образование', 'хорошее техническое образование']),
             'заряженные на успех студенты', 'практическое образование'])),
    'ВУЗ:РЭШ': If(AND([OR(['добротное изучение иностранных языков', 'хорошее математическое образование',
                           'хорошее экономическое образование']), 'академическое образование', 'внеучебные активности',
                       'заряженные на успех студенты', 'международное взаимодействие', 'нет военной кафедры',
                       'престиж'])),
}


class KnowledgeBase:
    def __init__(self, rules):
        self.rules = rules
        self.memory = {}

    def get(self, name):
        global queries_history
        if name in self.memory.keys():
            return self.memory[name]
        for fld in self.rules.keys():
            if fld == name or fld.startswith(name + ":"):
                # print(" + proving {}".format(fld))
                value = 'y' if fld == name else fld.split(':')[1]
                res = self.eval(self.rules[fld], field=name)
                if res == 'y':
                    self.memory[name] = value
                    return value
        # field is not found, using default
        res = self.eval(self.rules['default'], field=name)
        self.memory[name] = res
        return res

    def eval(self, expr, field=None):
        if isinstance(expr, Ask):
            if field == "ВУЗ":
                raise NoMoreUnivers
            elif field in self.rules:
                return 'n'
            if field not in queries_history:
                queries_history[field] = None
                print(field)
                return expr.ask()
            else:
                return queries_history[field]
        elif isinstance(expr, If):
            return self.eval(expr.x)
        elif isinstance(expr, AND) or isinstance(expr, list):
            expr = expr.x if isinstance(expr, AND) else expr
            for x in expr:
                if self.eval(x) == 'n':
                    return 'n'
            return 'y'
        elif isinstance(expr, OR):
            for x in expr.x:
                if self.eval(x) == 'y':
                    return 'y'
            return 'n'
        elif isinstance(expr, str):
            return self.get(expr)
        else:
            print("Unknown expr: {}".format(expr))


if __name__ == "__main__":
    universities = []
    try:
        while 3:
            kb = KnowledgeBase(rules)
            universities.append(kb.get('ВУЗ'))
            rules.pop(f"ВУЗ:{universities[-1]}")
    except NoMoreUnivers:
        if len(universities):
            print("Рекомендуем следующие университеты: ", universities)
        else:
            print("К сожалению, подходящий ВУЗ не найден :(")
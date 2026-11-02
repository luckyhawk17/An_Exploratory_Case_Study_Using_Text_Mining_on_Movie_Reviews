
class morphological_analysis:

    def __init__(self):
        pass    

    def bubble_sort(list):

        for i in range(len(list)-1): 

            for j in range(len(list)-2, i-1, -1):

                if list[j+1][1] > list[j][1]:
                    tmp = list[j]
                    list[j] = list[j+1]
                    list[j+1] = tmp
        
        return list

    
    def generate_word_list(morphoed_text):      
        
        flag = False

        w_list = []

        init_w = []
        init_w.append("") 
        init_w.append("") 
        w_list.append(init_w)        

        for word in morphoed_text:

            for w_list_e in w_list:                

                if word[0] == w_list_e[0] and word[1] == w_list_e[1]:
                    flag = True
                    break
            
            if flag == True:
                flag = False
                continue      

            w_list.append(word)
        
        del w_list[0:1]

        return w_list
    


        



           





    







    
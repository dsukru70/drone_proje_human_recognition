# drone_proje_human_recognition

**Yapılacak olan projede bireyler arası sosyal mesafe ölçümünü yapabilmek için bir görüntüleme cihazına sahip Drone’dan yararlanılacaktır. Drone üzerindeki kameralardan alınan görüntü, ön işleme, özellik çıkarımı, analiz ve görselleştirme adımları takip edilerek, mesafe ölçümü yapılabilecek çıktıya dönüştürülecektir.

Görüntü işleme süreci, insanların ve diğer objelerin tespit edilerek ayrılması ve insanların birbirleri arasındaki mesafenin ölçümünün yapılması şeklinde iki basamaktan oluşacaktır. Çalışmamız kapsamında insanların tespiti için alınan görüntülerin işlenmesi için OPENCV kütüphanesinin kullanılması planlanmaktadır. İnsanlar arası mesafe ölçümü için de yine NUMPY vb. matematiksel kütüphanelerden faydalanılacaktır (Hacıfazlıoğlu et al., n.d.). Hazırlanacak olan yazılım içerisinde kullanılacak olan fonksiyonel kütüphaneler ve geliştirme ortamı açık kaynak kodlu olarak tercih edilecektir.

Görüntü sayısal ve analog olmak üzere ikiye ayrılır. Drone’dan analog görüntü aktarımı olacağından, bilgisayar ortamında işleyebilmek için öncelikle sayısal görüntüye çevrilecektir. Sayısal görüntü, bir görüntünün en temel parçası olan piksel olarak adlandırılır. Piksel küçük kareler şeklindedir ve bu kareler birleşerek görüntüyü oluştururlar (Duman, n.d.). Bu işlem kenar belirleme olarak adlandırılır. Kenar belirleme işlemi, görüntüde bulunan insanların sınır piksellerini tespit etmek için bu piksellerin BCR değerlerinin değiştirilerek renklendirilmesi ile yapılır. Bireyler Drone den uzaklaştıkça belirlenen renkteki bireyleri içine almış kare küçülür ve yeni bireyleri tanımlamaya çalışır (Özer et al., 2022). Çalışmamızda görüntüyü gürültülerden arındırmak ve sayısallaştırma yapmak amacıyla öncelikle filtreleme işlemi yapılacaktır. Bu amaçla görüntüyü oluşturan her pikselin 0 ila 255 arası değer alan BCR değerleri düzenlenecektir.  Elde edilen temizlenmiş sayısal görüntüde, insan tespitinin yapılması için insana ait sınırların belirlenmesi işlemi gerçekleştirilecektir. Daha sonrasında ise insanlar arası ölçüm işlemi gerçekleştirilecektir      (KABUL GÖRMEYEN PROJEMİZDEN ALINTIDIR)**

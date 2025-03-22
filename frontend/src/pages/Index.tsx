import React, { useState, useEffect } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { 
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import Layout from "@/components/Layout";
import WaveformVisualizer from "@/components/WaveformVisualizer";
import { Mic, Play, Pause, Download, AudioWaveform, LoaderCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  audio_url: string | null;
  video_url: string | null;
  error: string | null;
}

const Index = () => {
  const [topic, setTopic] = useState("");
  const [additionalDetails, setAdditionalDetails] = useState("");
  const [waveformColor, setWaveformColor] = useState("blue");
  const [generatingPodcast, setGeneratingPodcast] = useState(false);
  const [podcastGenerated, setPodcastGenerated] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  
  const colorMap: Record<string, string> = {
    blue: "#1E90FF",
    purple: "#8A2BE2",
    pink: "#FF69B4",
    orange: "#FFA500",
    green: "#00FF00",
    red: "#FF0000"
  };
  
  const handleGeneratePodcast = async () => {
    if (!topic.trim()) {
      toast.error("Please enter a topic");
      return;
    }
    
    setGeneratingPodcast(true);
    setPodcastGenerated(false);
    setJobId(null);
    setJobStatus(null);
    setProgress(0);
    
    try {
      const response = await fetch("http://localhost:8000/api/podcast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          topic: topic,
          waveform_color: colorMap[waveformColor] || "#00FF00"
        }),
      });
      
      if (!response.ok) {
        throw new Error("Failed to start podcast generation");
      }
      
      const data = await response.json();
      setJobId(data.job_id);
      toast.success("Podcast generation started");
    } catch (error) {
      console.error("Error starting podcast generation:", error);
      toast.error("Failed to start podcast generation");
      setGeneratingPodcast(false);
    }
  };
  
  useEffect(() => {
    if (!jobId) return;
    
    const checkStatus = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/podcast/${jobId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch job status");
        }
        
        const data: JobStatus = await response.json();
        setJobStatus(data);
        setProgress(data.progress * 100);
        
        if (data.status === "completed") {
          setGeneratingPodcast(false);
          setPodcastGenerated(true);
          setAudioUrl(data.audio_url);
          toast.success("Podcast generated successfully");
        } else if (data.status === "failed") {
          setGeneratingPodcast(false);
          toast.error(`Podcast generation failed: ${data.error || "Unknown error"}`);
        } else {
          setTimeout(checkStatus, 2000);
        }
      } catch (error) {
        console.error("Error checking job status:", error);
        toast.error("Failed to check job status");
        setGeneratingPodcast(false);
      }
    };
    
    checkStatus();
  }, [jobId]);
  
  const togglePlayPause = () => {
    if (!podcastGenerated) return;
    setIsPlaying(!isPlaying);
  };
  
  const downloadPodcast = () => {
    if (!podcastGenerated || !jobStatus?.video_url) return;
    
    const a = document.createElement("a");
    a.href = `http://localhost:8000${jobStatus.video_url}`;
    a.download = `podcast_${topic.replace(/\s+/g, "_")}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    toast.success("Podcast downloading...");
  };
  
  return (
    <Layout>
      <div className="max-w-4xl mx-auto w-full space-y-8 animate-fade-in">
        <section className="text-center space-y-3">
          <h1 className="text-4xl font-bold tracking-tight">AI Podcast Generator</h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Turn any topic into a professional-sounding podcast with AI-generated audio and
            customizable waveform visualizations.
          </p>
        </section>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card className="overflow-hidden border-border/30 bg-background/70 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5" />
                Create Your Podcast
              </CardTitle>
              <CardDescription>
                Fill in the details to generate your AI podcast
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="topic">Topic</Label>
                <Input
                  id="topic"
                  placeholder="Enter your podcast topic"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="details">Additional Details (optional)</Label>
                <Textarea
                  id="details"
                  placeholder="Add more context, specific points to cover, etc."
                  value={additionalDetails}
                  onChange={(e) => setAdditionalDetails(e.target.value)}
                  rows={4}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="waveform-color">Waveform Color</Label>
                <Select
                  value={waveformColor}
                  onValueChange={setWaveformColor}
                >
                  <SelectTrigger id="waveform-color">
                    <SelectValue placeholder="Select color" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="blue">Blue</SelectItem>
                    <SelectItem value="purple">Purple</SelectItem>
                    <SelectItem value="pink">Pink</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="green">Green</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
            <CardFooter>
              <Button 
                className="w-full gap-2"
                variant="orange"
                disabled={generatingPodcast || !topic.trim()}
                onClick={handleGeneratePodcast}
              >
                {generatingPodcast ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <AudioWaveform className="h-4 w-4" />
                    Generate Podcast
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
          
          <Card className="overflow-hidden border-border/30 bg-background/70 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AudioWaveform className="h-5 w-5" />
                Podcast Preview
              </CardTitle>
              <CardDescription>
                {podcastGenerated 
                  ? "Your generated podcast is ready to play"
                  : "Your podcast will appear here after generation"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {generatingPodcast && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{jobStatus?.status || "Processing..."}</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
              
              <div className={cn(
                "transition-opacity duration-300",
                podcastGenerated ? "opacity-100" : "opacity-50",
              )}>
                <WaveformVisualizer 
                  playing={isPlaying}
                  color={waveformColor}
                />
              </div>
              
              {podcastGenerated ? (
                <div className="text-center">
                  <h3 className="font-medium text-lg">{topic}</h3>
                  <p className="text-sm text-muted-foreground">
                    AI-generated podcast
                  </p>
                </div>
              ) : (
                <div className="flex items-center justify-center h-20">
                  {generatingPodcast ? (
                    <div className="text-center space-y-2">
                      <LoaderCircle className="h-8 w-8 animate-spin mx-auto text-primary/70" />
                      <p className="text-sm text-muted-foreground">
                        Crafting your podcast...
                      </p>
                    </div>
                  ) : (
                    <p className="text-muted-foreground text-center">
                      Fill in the details and generate your podcast to see a preview
                    </p>
                  )}
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between gap-4">
              <Button
                variant="outline"
                className="flex-1 gap-2"
                disabled={!podcastGenerated}
                onClick={togglePlayPause}
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-4 w-4" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Play
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                className="flex-1 gap-2"
                disabled={!podcastGenerated}
                onClick={downloadPodcast}
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            </CardFooter>
          </Card>
        </div>
        
        <div className="text-center text-sm text-muted-foreground">
          <p>
            Note: This application connects to a local API server running at http://localhost:8000.
            Make sure the server is running before generating podcasts.
          </p>
        </div>
      </div>
    </Layout>
  );
};

export default Index;

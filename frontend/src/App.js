import { useEffect, useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import axios from "axios";
import { BarChart3, TrendingUp, Users, DollarSign, LineChart, Filter, RefreshCw, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DashboardLayout = ({ children }) => {
  const location = useLocation();
  const [isGenerating, setIsGenerating] = useState(false);

  const navItems = [
    { path: "/", label: "Overview", icon: BarChart3 },
    { path: "/statistical", label: "Statistical Tests", icon: TrendingUp },
    { path: "/segments", label: "Segment Analysis", icon: Filter },
    { path: "/funnel", label: "Funnel", icon: Users },
    { path: "/insights", label: "Business Insights", icon: FileText },
  ];

  const handleGenerateData = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API}/experiment/generate`);
      toast.success("Dataset generated successfully! 20,000 users created.");
      window.location.reload();
    } catch (error) {
      toast.error("Failed to generate dataset");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA]">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-[#064E3B]" data-testid="app-title">A/B Experiment Analytics</h1>
              <p className="text-sm text-slate-600 mt-0.5">Premium Subscription CTA Optimization</p>
            </div>
            <Button 
              onClick={handleGenerateData} 
              disabled={isGenerating}
              className="bg-[#064E3B] hover:bg-[#064E3B]/90 text-white font-medium px-6 py-2.5 rounded-lg shadow-sm transition-all active:scale-95"
              data-testid="generate-data-btn"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isGenerating ? 'animate-spin' : ''}`} />
              {isGenerating ? 'Generating...' : 'Regenerate Data'}
            </Button>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
                  className={`flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? 'border-[#4F46E5] text-[#4F46E5]'
                      : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {children}
      </main>
    </div>
  );
};

const MetricCard = ({ title, value, subtitle, icon: Icon, trend, trendValue, colorClass = "text-[#064E3B]" }) => (
  <Card className="bg-white border border-slate-200 shadow-sm rounded-xl hover:shadow-md transition-shadow duration-200" data-testid={`metric-card-${title.toLowerCase().replace(/\s+/g, '-')}`}>
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <CardTitle className="text-sm font-medium text-slate-600 uppercase tracking-wider">{title}</CardTitle>
        {Icon && <Icon className={`w-5 h-5 ${colorClass}`} />}
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-1">
        <div className={`text-3xl font-semibold tracking-tight ${colorClass}`} data-testid={`metric-value-${title.toLowerCase().replace(/\s+/g, '-')}`}>{value}</div>
        {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
        {trend && (
          <div className={`flex items-center text-sm font-medium ${
            trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-slate-600'
          }`}>
            <TrendingUp className="w-3 h-3 mr-1" />
            {trendValue}
          </div>
        )}
      </div>
    </CardContent>
  </Card>
);

const Overview = () => {
  const [config, setConfig] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [timeseries, setTimeseries] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [configRes, metricsRes, timeseriesRes] = await Promise.all([
          axios.get(`${API}/experiment/config`),
          axios.get(`${API}/metrics/summary`),
          axios.get(`${API}/analysis/timeseries`)
        ]);
        setConfig(configRes.data);
        setMetrics(metricsRes.data);
        setTimeseries(timeseriesRes.data.timeseries);
      } catch (error) {
        if (error.response?.status === 404) {
          toast.error("No data found. Please generate data first.");
        }
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64" data-testid="loading-spinner"><div className="text-slate-600">Loading...</div></div>;
  }

  if (!config || !metrics) {
    return (
      <div className="text-center py-12" data-testid="no-data-message">
        <p className="text-slate-600 mb-4">No experiment data available. Please generate data first.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="overview-page">
      {/* Experiment Info */}
      <Card className="bg-white border border-slate-200 shadow-sm rounded-xl">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-[#064E3B]">Experiment Configuration</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Hypothesis</h4>
              <p className="text-base text-slate-700 leading-relaxed">{config.hypothesis}</p>
            </div>
            <div className="space-y-3">
              <div>
                <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Control</h4>
                <p className="text-sm text-slate-700">{config.control_variant}</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Treatment</h4>
                <p className="text-sm text-slate-700">{config.treatment_variant}</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Sample Size</h4>
                <p className="text-sm text-slate-700">{config.sample_size.toLocaleString()} users</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Conversion Rate (Control)"
          value={`${(metrics.conversion_rate_control * 100).toFixed(2)}%`}
          icon={Users}
          colorClass="text-slate-700"
        />
        <MetricCard
          title="Conversion Rate (Treatment)"
          value={`${(metrics.conversion_rate_treatment * 100).toFixed(2)}%`}
          icon={Users}
          trend="up"
          trendValue={`+${metrics.relative_uplift.toFixed(1)}% vs control`}
          colorClass="text-[#4F46E5]"
        />
        <MetricCard
          title="Revenue Per User (Treatment)"
          value={`$${metrics.revenue_per_user_treatment.toFixed(2)}`}
          subtitle={`Control: $${metrics.revenue_per_user_control.toFixed(2)}`}
          icon={DollarSign}
          colorClass="text-[#10B981]"
        />
        <MetricCard
          title="Relative Uplift"
          value={`${metrics.relative_uplift > 0 ? '+' : ''}${metrics.relative_uplift.toFixed(1)}%`}
          subtitle={`Absolute: ${metrics.absolute_uplift > 0 ? '+' : ''}${metrics.absolute_uplift.toFixed(2)}pp`}
          icon={TrendingUp}
          trend={metrics.relative_uplift > 0 ? 'up' : 'down'}
          colorClass="text-[#064E3B]"
        />
      </div>

      {/* Time Series */}
      {timeseries && timeseries.length > 0 && (
        <Card className="bg-white border border-slate-200 shadow-sm rounded-xl">
          <CardHeader className="border-b border-slate-100 pb-4">
            <CardTitle className="text-xl font-semibold text-[#064E3B]">Daily Performance Trends</CardTitle>
            <CardDescription>Conversion rate and CTR over experiment duration</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <Tabs defaultValue="conversion" className="w-full">
              <TabsList className="mb-4">
                <TabsTrigger value="conversion" data-testid="tab-conversion">Conversion Rate</TabsTrigger>
                <TabsTrigger value="ctr" data-testid="tab-ctr">Click-Through Rate</TabsTrigger>
              </TabsList>
              <TabsContent value="conversion">
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsLineChart data={timeseries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="day" label={{ value: 'Experiment Day', position: 'insideBottom', offset: -5 }} />
                    <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                    <Legend />
                    <Line type="monotone" dataKey="control_conversion" stroke="#64748B" strokeWidth={2} name="Control" />
                    <Line type="monotone" dataKey="treatment_conversion" stroke="#4F46E5" strokeWidth={2} name="Treatment" />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </TabsContent>
              <TabsContent value="ctr">
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsLineChart data={timeseries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="day" label={{ value: 'Experiment Day', position: 'insideBottom', offset: -5 }} />
                    <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                    <Legend />
                    <Line type="monotone" dataKey="control_ctr" stroke="#64748B" strokeWidth={2} name="Control" />
                    <Line type="monotone" dataKey="treatment_ctr" stroke="#4F46E5" strokeWidth={2} name="Treatment" />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const StatisticalTests = () => {
  const [tests, setTests] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API}/analysis/statistical-tests`);
        setTests(response.data.tests);
      } catch (error) {
        toast.error("Failed to load statistical tests");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64" data-testid="loading-spinner"><div className="text-slate-600">Loading...</div></div>;
  }

  if (!tests) return null;

  return (
    <div className="space-y-6" data-testid="statistical-tests-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#064E3B] mb-2">Statistical Analysis</h2>
        <p className="text-base text-slate-600">Comprehensive statistical tests to validate experiment results</p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {tests.map((test, idx) => (
          <Card key={idx} className="bg-white border border-slate-200 shadow-sm rounded-xl" data-testid={`test-card-${idx}`}>
            <CardHeader className="border-b border-slate-100 pb-4">
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-lg font-semibold text-[#064E3B]">{test.test_name}</CardTitle>
                  <CardDescription className="mt-2">{test.interpretation}</CardDescription>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                  test.significant ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-700'
                }`}>
                  {test.significant ? 'Significant' : 'Not Significant'}
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Test Statistic</h4>
                  <p className="text-xl font-mono text-[#4F46E5]" data-testid={`test-statistic-${idx}`}>{test.statistic.toFixed(4)}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">P-Value</h4>
                  <p className="text-xl font-mono text-[#4F46E5]" data-testid={`test-pvalue-${idx}`}>
                    {test.p_value < 0.001 ? '<0.001' : test.p_value.toFixed(4)}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">95% Confidence Interval</h4>
                  <p className="text-sm font-mono text-slate-700" data-testid={`test-ci-${idx}`}>
                    [{test.confidence_interval[0].toFixed(2)}, {test.confidence_interval[1].toFixed(2)}]
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

const SegmentAnalysis = () => {
  const [segments, setSegments] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API}/analysis/segments`);
        setSegments(response.data.segments);
      } catch (error) {
        toast.error("Failed to load segment data");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64" data-testid="loading-spinner"><div className="text-slate-600">Loading...</div></div>;
  }

  if (!segments) return null;

  const groupedSegments = segments.reduce((acc, seg) => {
    if (!acc[seg.segment_name]) acc[seg.segment_name] = [];
    acc[seg.segment_name].push(seg);
    return acc;
  }, {});

  return (
    <div className="space-y-6" data-testid="segment-analysis-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#064E3B] mb-2">Segment Analysis</h2>
        <p className="text-base text-slate-600">Performance breakdown by device, region, and user tenure</p>
      </div>

      {Object.entries(groupedSegments).map(([segmentName, segmentData]) => (
        <Card key={segmentName} className="bg-white border border-slate-200 shadow-sm rounded-xl" data-testid={`segment-${segmentName.toLowerCase().replace(/\s+/g, '-')}`}>
          <CardHeader className="border-b border-slate-100 pb-4">
            <CardTitle className="text-xl font-semibold text-[#064E3B]">{segmentName}</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Segment</th>
                    <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Control Conv.</th>
                    <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Treatment Conv.</th>
                    <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Uplift</th>
                    <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Sample Size</th>
                  </tr>
                </thead>
                <tbody>
                  {segmentData.map((seg, idx) => (
                    <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50 transition-colors" data-testid={`segment-row-${idx}`}>
                      <td className="py-3 text-sm font-medium text-slate-700">{seg.segment_value}</td>
                      <td className="py-3 text-sm text-right font-mono text-slate-600">{(seg.control_conversion * 100).toFixed(2)}%</td>
                      <td className="py-3 text-sm text-right font-mono text-[#4F46E5] font-medium">{(seg.treatment_conversion * 100).toFixed(2)}%</td>
                      <td className={`py-3 text-sm text-right font-mono font-medium ${
                        seg.uplift > 0 ? 'text-green-600' : seg.uplift < 0 ? 'text-red-600' : 'text-slate-600'
                      }`}>
                        {seg.uplift > 0 ? '+' : ''}{seg.uplift.toFixed(1)}%
                      </td>
                      <td className="py-3 text-sm text-right text-slate-600">{seg.sample_size.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

const FunnelAnalysis = () => {
  const [funnel, setFunnel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API}/analysis/funnel`);
        setFunnel(response.data.funnel);
      } catch (error) {
        toast.error("Failed to load funnel data");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64" data-testid="loading-spinner"><div className="text-slate-600">Loading...</div></div>;
  }

  if (!funnel) return null;

  const chartData = funnel.map(step => ({
    name: step.step,
    Control: Number((step.control_rate * 100).toFixed(1)),
    Treatment: Number((step.treatment_rate * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6" data-testid="funnel-analysis-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#064E3B] mb-2">Conversion Funnel</h2>
        <p className="text-base text-slate-600">Step-by-step comparison of user journey</p>
      </div>

      <Card className="bg-white border border-slate-200 shadow-sm rounded-xl">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-[#064E3B]">Funnel Comparison</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis type="number" tickFormatter={(value) => `${value}%`} />
              <YAxis type="category" dataKey="name" width={120} />
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
              <Bar dataKey="Control" fill="#64748B" radius={[0, 8, 8, 0]} />
              <Bar dataKey="Treatment" fill="#4F46E5" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="bg-white border border-slate-200 shadow-sm rounded-xl">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-[#064E3B]">Detailed Funnel Metrics</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Funnel Step</th>
                  <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Control Count</th>
                  <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Treatment Count</th>
                  <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Control Rate</th>
                  <th className="text-right text-sm font-medium text-slate-500 uppercase tracking-wider pb-3">Treatment Rate</th>
                </tr>
              </thead>
              <tbody>
                {funnel.map((step, idx) => (
                  <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50 transition-colors" data-testid={`funnel-row-${idx}`}>
                    <td className="py-3 text-sm font-medium text-slate-700">{step.step}</td>
                    <td className="py-3 text-sm text-right font-mono text-slate-600">{step.control_count.toLocaleString()}</td>
                    <td className="py-3 text-sm text-right font-mono text-[#4F46E5]">{step.treatment_count.toLocaleString()}</td>
                    <td className="py-3 text-sm text-right font-mono text-slate-600">{(step.control_rate * 100).toFixed(2)}%</td>
                    <td className="py-3 text-sm text-right font-mono text-[#4F46E5] font-medium">{(step.treatment_rate * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

const BusinessInsights = () => {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API}/recommendations`);
        setRecommendations(response.data.recommendations);
      } catch (error) {
        toast.error("Failed to load recommendations");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64" data-testid="loading-spinner"><div className="text-slate-600">Loading...</div></div>;
  }

  if (!recommendations) return null;

  const priorityColors = {
    High: 'bg-red-100 text-red-700 border-red-200',
    Medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    Low: 'bg-blue-100 text-blue-700 border-blue-200',
  };

  return (
    <div className="space-y-6" data-testid="business-insights-page">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#064E3B] mb-2">Business Insights & Recommendations</h2>
        <p className="text-base text-slate-600">Actionable recommendations for product stakeholders</p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {recommendations.map((rec, idx) => (
          <Card key={idx} className="bg-white border border-slate-200 shadow-sm rounded-xl hover:shadow-md transition-shadow" data-testid={`recommendation-card-${idx}`}>
            <CardHeader className="border-b border-slate-100 pb-4">
              <div className="flex items-start justify-between">
                <CardTitle className="text-lg font-semibold text-[#064E3B]">{rec.title}</CardTitle>
                <div className={`px-3 py-1 rounded-lg text-xs font-medium border ${priorityColors[rec.priority]}`}>
                  {rec.priority} Priority
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Analysis</h4>
                  <p className="text-base text-slate-700 leading-relaxed">{rec.description}</p>
                </div>
                <div className="bg-[#F0FDF4] border border-[#10B981]/20 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-[#10B981] uppercase tracking-wider mb-1">Expected Impact</h4>
                  <p className="text-sm text-slate-700">{rec.expected_impact}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/statistical" element={<StatisticalTests />} />
            <Route path="/segments" element={<SegmentAnalysis />} />
            <Route path="/funnel" element={<FunnelAnalysis />} />
            <Route path="/insights" element={<BusinessInsights />} />
          </Routes>
        </DashboardLayout>
      </BrowserRouter>
    </div>
  );
}

export default App;